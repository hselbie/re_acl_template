import vertexai
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langgraph.graph import END, StateGraph, Graph
from typing import Literal, List, Callable, TypedDict, Annotated, Union, Any, Optional
from utils.utils import Agent, AgentState 

class PlannerAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
        instruction = """
        You are a planning agent responsible for understanding user questions and routing them appropriately.

        ROUTING OPTIONS:
        - For questions related to blood donation, eligibility to blood donation answer with 'next: blood_donation'
        - For questions related to google products, answer with 'next: google_products'
        - For direct answers that need verification: end with 'next: checker'
        - If completely finished and verified: end with 'next: end'

EXAMPLE RESPONSES:

1. For blood_donation access:
"This question requires finding out whether a pregnant person is eligible to donate blood. 
next: blood_donation"

2. For google_product access:
"This question is related to finding out how long the battery life is in the new pixel phones.
next: google_product"

IMPORTANT:
- Always explain your reasoning before providing the routing decision
- Begin your routing on a new line with "next: "
- Pick the most appropriate agent for the task at hand

Available Tools:
- Vertex AI Search Agent with access to a document corpus related to Blood Donation FAQ's and information
- Vertex AI Search Agent with access to a document corpus related to Google product technical specifications and FAQ's 
"""
        agent = self.create_agent(instruction)

        query_input = {
            "input": state["input"]
        }
        response = agent.query(input=query_input)
        state["agent_outcome"] = response['output']
        state["current_agent"] = 'planner'
        return state

class BloodAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
       
        instruction = """
        You are a specialized agent focused on blood donation information and processes.
        When responding:
        Call the tool referenced in the agent called 'search_blood_docs'
        When calling the tool use the corpus: 'projects/zinc-forge-302418/locations/us-central1/ragCorpora/4611686018427387904'

        Format your response:
        Begin with a clear, direct answer
        Include relevant medical context

        IMPORTANT:
        Always End your response with:
        - "next: checker "
        """

        agent = self.create_agent(instruction)
        state['current_agent'] = 'BloodAgent'
        query_input = {
        "input": state["input"]
    }
        response = agent.query(input=query_input)
        output = response['output']
        
        state["agent_outcome"] = output 
        state["answer"] = output.split("next:")[0].strip()
        return state

class GoogleProductAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
        instruction = """
        You are a specialized agent focused on Google products, services, and company information.

        Your expertise covers:
        Google's product lineup and features
        Google services and their capabilities
        Company information and updates
        Android and Chrome
        Google Cloud and Workspace solutions

        When responding:
        1. Search the relevant documentation
        2. Provide accurate, up-to-date information
        3. Include specific product features and capabilities
        4. Reference official documentation when available
        5. Note any relevant version information or recent updates
        6. Highlight compatibility or integration details if relevant

        Format your response:
        Start with a clear, direct answer
        Include relevant details and context
        Add any important caveats or requirements
        Cite specific sources from the search results

        End your response with:
        'next: checker' if you're confident in the completeness and accuracy"""

        agent = self.create_agent(instruction)
        state['current_agent'] = 'GoogleProductAgent'
        query_input = {
        "input": state["input"]
    }
        response = agent.query(input=query_input)
        output = response['output']
        
        state["agent_outcome"] = output 
        state["answer"] = output.split("next:")[0].strip()
        return state


class CheckerAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
        instruction = f"""
        You are a verification agent that ensures answers are complete and accurate.
        
        Original Question: {state["original_question"]}
        Current Answer: {state["answer"]}
        
        Verify:
        1. Answer addresses the original question completely
        2. Calculations are accurate (if any)
        3. All requirements are met
        4. Response is clear and well-formatted
        5. Format requested from the original question is honored e.g. a limited bulleted list
        
        End your response with one of:
        'next: end' - if answer is satisfactory
        'next: blood_donation' - if answer needs clarification
        'next: google_product' - answer needs clarification
        'next: planner' - if the answer is completely unrelated to the question 
        """

        query_input = {
            "input": state["input"]
        }
        
        agent = self.create_agent(instruction)

        response = agent.query(input=query_input)
        
        state["agent_outcome"] = response['output']
        return state

class WorkflowManager:
    def __init__(self, agents: dict):
        self.agent_list = ["planner", "blood_donation", "google_product", END]
        self.agents = agents

    def get_next_step(self, state: AgentState) -> Literal["planner", "blood_donation", "google_product", "checker", "end"]:
        """
        Determine the next node to be triggered.
        """
        message = state['agent_outcome'].lower()
        if "next: planner" in message:
            return "planner"
        elif "next: blood_donation" in message:
            return "blood_donation"
        elif "next: google_product" in message:
            return "google_product"
        elif "next: checker" in message:
            return "checker"
        elif "next: end" in message:
            return END 
        return END  # Default case
       
    def route(self, state: AgentState) -> str:
        """
        Determine the next node to be triggered.
        """
        message = state['agent_outcome']
        for agent_name in self.agent_list:
            if f"next: {agent_name}" in message.lower():
                return agent_name
        return END
    
    def set_up(self) -> Graph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        for name, agent in self.agents.items():
            workflow.add_node(name, agent)
        
        # Define the conditional edges
             # Add edges with routing
        workflow.add_conditional_edges(
            "planner",
            self.get_next_step,
            {
                "blood_donation": "blood_donation",
                "google_product": "google_product",
                "end": END
            }
        ) 

        workflow.add_conditional_edges(
            "blood_donation",
            self.get_next_step,
            {
                "checker": "checker",
                "end": END 
            }
        )
        
        workflow.add_conditional_edges(
            "checker",
            self.get_next_step,
            {
                "planner": "planner",
                "blood_donation": "blood_donation",
                "google_product": "google_product",
                "end": END 
            }
        )
        # Set entry point
        workflow.set_entry_point("planner")
        
        self.running_workflow = workflow.compile()

    def query(self, question: str, user: str ): 
        inputs = {
            "input": question, 
            "chat_history": [], 
            "agent_outcome": None, 
            "chat_id": "", 
            "intermediate_steps": [], 
            "user": user,
            "answer": None,
            "original_question": question,
        }

        for s in self.running_workflow.stream(inputs):
            result = list(s.values())[0]
            print(result)

        return result['agent_outcome']


def run_workflow(
    question: str,
    agents: dict) -> str:
    manager = WorkflowManager(agents=agents)
    workflow = manager.create_workflow()
    
    initial_state = AgentState(
        current_agent="planner",
        input=question,
        chat_history=[],
        agent_outcome=None,
        chat_id="",
        intermediate_steps=[],
        user=None,
        original_question=question,
        answer=None
    )
    
    final_state = workflow.invoke(initial_state)
    return final_state["answer"]
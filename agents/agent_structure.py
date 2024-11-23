import vertexai
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langgraph.graph import END, StateGraph, Graph
from typing import Literal, List, Callable, TypedDict, Annotated, Union, Any, Optional
from utils.utils import Agent, AgentState 

class PlannerAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
        instruction = """You are a planning agent responsible for understanding user questions and routing them appropriately.

ROUTING OPTIONS:
- For calculations or data processing: end with 'next: adder'
- For direct answers that need verification: end with 'next: checker'
- If completely finished and verified: end with 'next: end'

EXAMPLE RESPONSES:

1. For calculation needed:
"This question requires calculating multiple steps with apples. Let me route this to our calculation specialist.
next: adder"

2. For direct answer:
"Paris is the capital of France. This is a straightforward fact that needs verification.
next: checker"

3. For complex processing:
"This question involves analyzing sales data across multiple quarters and calculating growth rates. This needs our processing agent.
next: adder"

IMPORTANT:
- Always explain your reasoning before providing the routing decision
- Begin your routing on a new line with "next: "
- Pick the most appropriate agent for the task at hand

Available Tools:
- Calculator for basic math
- Search for factual information
"""
        
        agent = self.create_agent(instruction)

        query_input = {
            "input": state["input"]
        }
        response = agent.query(input=query_input)
        state["agent_outcome"] = response['output']
        state["current_agent"] = 'planner'
        if "next: adder" not in response['output']:
            state["answer"] = response.split("next:")[0].strip()
            
        return state

class AdderAgent(Agent):
    def __call__(self, state: AgentState) -> AgentState:
        instruction = f"""You are a processing agent that handles complex calculations and detailed analysis.
        
        Original Question: {state["original_question"]}
        
        Your task:
        1. Process the question using available tools
        2. Show your work clearly
        3. Provide a detailed answer
        
        Always end your response with: 'next: checker'
        """
        
        agent = self.create_agent(instruction)
        state['current_agent'] = 'adder'
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
        instruction = f"""You are a verification agent that ensures answers are complete and accurate.
        
        Original Question: {state["original_question"]}
        Current Answer: {state["answer"]}
        
        Verify:
        1. Answer addresses the original question completely
        2. Calculations are accurate (if any)
        3. All requirements are met
        4. Response is clear and well-formatted
        
        End your response with one of:
        'next: end' - if answer is satisfactory
        'next: adder' - if calculations need revision
        'next: planner' - if answer needs clarification
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
        self.agent_list = ["planner", "adder", "checker", END]
        self.agents = agents

    def get_next_step(self, state: AgentState) -> Literal["planner", "adder", "checker", "end"]:
        """
        Determine the next node to be triggered.
        """
        message = state['agent_outcome'].lower()
        if "next: planner" in message:
            return "planner"
        elif "next: adder" in message:
            return "adder"
        elif "next: checker" in message:
            return "checker"
        elif "next: end" in message:
            return "end"
        return "end"  # Default case
       
    def route(self, state: AgentState) -> str:
        """
        Determine the next node to be triggered.
        """
        message = state['agent_outcome']
        for agent_name in self.agent_list:
            if f"next: {agent_name}" in message.lower():
                return agent_name
        return END
    
    def create_workflow(self) -> Graph:
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
                "adder": "adder",
                "checker": "checker",
                "end": END
            }
        ) 

        workflow.add_conditional_edges(
            "adder",
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
                "adder": "adder",
                "end": END 
            }
        )
        # Set entry point
        workflow.set_entry_point("planner")
        
        return workflow.compile()

def run_workflow(
    question: str,
    agents: dict) -> str:
    manager = WorkflowManager(agents=agents)
    workflow = manager.create_workflow()
    
    initial_state = AgentState(
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
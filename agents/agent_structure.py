import vertexai
from vertexai.preview import reasoning_engines
from langchain_core.tools import tool
from langchain_core import prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from typing import Literal, List, Callable, TypedDict, Annotated, Union, Any, Optional
from utils.utils import Agent, AgentState, GraphNode


class CustomAgent(Agent):
    def create_agent(self, instruction: str):
        vertexai.init(project=self.project, location=self.location)
        custom_prompt= {
            "user_input": lambda x: x["input"],
            # "history": lambda x: x["history"],
            "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
        } | ChatPromptTemplate.from_messages([
            ("system", instruction),
            # prompts.MessagesPlaceholder(variable_name="history")
            ("placeholder", "{history}"),
            ("user", "{user_input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = reasoning_engines.LangchainAgent(
            prompt=custom_prompt,
            model=self.model,
            model_kwargs=self.model_kwargs,
            tools=self.tools,
            agent_executor_kwargs={"return_intermediate_steps": False},
        )

        return agent

class MultiAgent:
    def __init__(self, project: str, agent_list: List[str], location: str, enable_tracing = False) -> None:
        self.project = project
        self.location = location
        self.agent_list = agent_list

        self.enable_tracing = enable_tracing

    def route(self, state: AgentState):
        """
        Route to determine the next node to be triggered.
        """
        message = state['agent_outcome']
        for agent_name in self.agent_list:
            if f"next: {agent_name}" in message:
                return agent_name
        return END

    def _setup(self):

        workflow = StateGraph(AgentState)
        # Add graph nodes
        workflow.add_node(node="coordinator_agent",
                          action=GraphNode("coordinator_agent", coordinator_agent)
        )
        workflow.add_node(node="rapida_machine_type_agent",
                          action=GraphNode("rapida_machine_type_agent", rapida_machine_type_agent)
        )
        workflow.add_node(node="rotajet_machine_type_agent",
                          action=GraphNode("rotajet_machine_type_agent", rotajet_machine_type_agent)
        )
        workflow.add_node(node="rapida_106_agent",
                          action=GraphNode("rapida_106_agent", rapida_106_agent)
        )
        workflow.add_node(node="rapida_75_agent",
                          action=GraphNode("rapida_75_agent", rapida_75_agent)
        )
        workflow.add_node(node="rotajet_2018_agent",
                          action=GraphNode("rotajet_2018_agent", rotajet_2018_agent)
        )

        # Set entry point
        workflow.set_entry_point("coordinator_agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            source="coordinator_agent",
            path=route
        )
        workflow.add_conditional_edges(
            source="rapida_machine_type_agent",
            path=route
        )
        workflow.add_conditional_edges(
            source="rotajet_machine_type_agent",
            path=route
        )

        # Add normal edges
        workflow.add_edge(
            start_key='rapida_106_agent',
            end_key=END
        )
        workflow.add_edge(
            start_key='rapida_75_agent',
            end_key=END
        )
        workflow.add_edge(
            start_key='rotajet_2018_agent',
            end_key=END
        )

        self.runnable_workflow = workflow.compile()
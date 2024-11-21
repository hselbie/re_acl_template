import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import tool
from typing import Literal, List, Callable, TypedDict, Annotated, Union, Any, Optional
import operator

class AgentState(TypedDict):
    """
    The agent state is the input to each node in the graph
    """
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    chat_id: str
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    user: Optional[dict]

class Agent:  # Base Agent class
    def __init__(self,model: str, project: str, location: str, state: AgentState=None, model_kwargs: dict=None,tools: Union[None, List[tool]] = None):
        self.model = model
        self.project = project
        self.location = location
        self.tools = tools
        self.model_safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        } 
        self.model_kwargs = model_kwargs if model_kwargs else {
            "temperature": 0.28,
            "max_output_tokens": 1000,
            "top_p": 0.95,
            "top_k": None,
            "safety_settings": self.model_safety_settings,
            }

    def update_state(self, key, value):
        self.state = {**self.state, key: value}

class GraphNode():
    """
       Node to be added to a graph.
    """
    def __init__(self, name: str, agent: Callable) -> None:
        self.name = name
        self.agent = agent

    def __call__(self, state) -> Any:
        print(f"{self.name} is triggered")

        # user_content = "\n".join(context["value"] for context in state["user"]["context_notes"] if context["type"] == "agent" and self.name == context["reference"])
    #     if user_content != "":
    #         user_content = "You are given with additional information:\n" + user_content
    #     dynamic_context = (
    #     "Please use time format as YYYY-MM-DD.\n"
    #    f"{user_content}"
    #     )

        # Invoke agent
        response = self.agent.query(input=state)
            # input={"input": state["input"]
            #        , "dynamic_context": dynamic_context},
            # config={"configurable": {"session_id": state["chat_id"]}
            # },
        print(f"{self.name} outcome:", response["output"])
        # return {"agent_outcome": response["output"], "intermediate_steps": [(state['agent_outcome'], response["output"])]}


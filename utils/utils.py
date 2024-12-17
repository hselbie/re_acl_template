from vertexai.preview import reasoning_engines
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.messages import BaseMessage
from langchain_core import prompts
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Literal, List, Callable, TypedDict, Annotated, Union, Any, Optional
import operator

class AgentState(TypedDict):
    """
    The agent state is the input to each node in the graph
    """
    original_question: str     # The original question asked
    input: str                  # Current input string being processed
    current_agent: str          # Name of current agent
    chat_history: list[BaseMessage]  # List of previous messages
    agent_outcome: Union[AgentAction, AgentFinish, None]  # Result of agent's action
    chat_id: str               # Unique identifier for chat session
    intermediate_steps: list[tuple[AgentAction, str]]  # Steps taken by agent
    user: Optional[dict]       # Optional user-related data
    original_question: str     # The initial question asked
    answer: Optional[str]      # The current answer being developed

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
        
    def create_agent(self, instruction: str):
        vertexai.init(project=self.project, location=self.location)
        custom_prompt = {
        "input": lambda x: x["input"],
        "agent_scratchpad": (lambda x: format_to_tool_messages(x["intermediate_steps"])),
        } | prompts.ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("user", "{input}"),
            prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
        # Create the full prompt with input mappings
        return reasoning_engines.LangchainAgent(
            prompt=custom_prompt,
            model=self.model,
            model_kwargs=self.model_kwargs,
            tools=self.tools,
            agent_executor_kwargs={"return_intermediate_steps": False},
            enable_tracing=True
        )

    def update_state(self, key, value):
        self.state = {**self.state, key: value}
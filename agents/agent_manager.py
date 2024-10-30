import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.tools.base import StructuredTool
from langchain_core import prompts
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from utils import create_logger

from typing import TypedDict, Callable, Sequence, Union, Annotated, Optional, List
import operator 

logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

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

class AGENT:
    def __init__(
            self,
            model: str,
            tools: Sequence[Callable],
            project: str,
            location: str,
        ):
        self.model_name = model
        self.tools = tools
        self.project = project
        self.location = location

    def set_up(self):
        """All unpickle-able logic should go here.

        The .set_up() method should not be called for an object that is being
        prepared for deployment.
        """
        vertexai.init(project=self.project, location=self.location)

        prompt = {
            "input": lambda x: x["input"],
            "agent_scratchpad": (
                lambda x: format_to_tool_messages(x["intermediate_steps"])
            ),
        } | prompts.ChatPromptTemplate.from_messages([
            ("user", "{input}"),
            prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatVertexAI(model_name=self.model_name)
        if self.tools:
            llm = llm.bind_tools(tools=self.tools)

        self.agent_executor = AgentExecutor(
            agent=prompt | llm | ToolsAgentOutputParser(),
            tools=[StructuredTool.from_function(tool) for tool in self.tools],
        )

    def query(self, input: str):
        """Query the application.

        Args:
            input: The user prompt.

        Returns:
            The output of querying the application with the given input.
        """
        response = self.agent_executor.invoke(input={"input": input})
        logger.info(response)
        return response




# Define a function to create agents with instruction and some tools
# def create_agent(instruction: str, tools: Union[None, List[tool]]):

#     # Define prompt template
#     prompt = {
#         # "history": lambda x: x["history"],
#         "input": lambda x: x["input"],
#         "agent_scratchpad": (lambda x: format_to_tool_messages(x["intermediate_steps"])),
#         "dynamic_context": lambda x: x["dynamic_context"],
#     } | prompts.ChatPromptTemplate.from_messages(
#         [
#             ("system", instruction.format(dynamic_context="{dynamic_context}")),
#             # prompts.MessagesPlaceholder(variable_name="history"),
#             ("user", "{input}"),
#             prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ]
#     )

#     agent = reasoning_engines.LangchainAgent(
#         prompt=prompt,
#         model=LLM_MODEL,
#         # chat_history=get_session_history,
#         model_kwargs={"temperature": 0},
#         tools=tools,
#         agent_executor_kwargs={"return_intermediate_steps": False},
#     )

#     return agent
from tools import tool_collection
from typing import Union, List
from langchain_core.tools import tool
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from vertexai.preview import reasoning_engines
from langchain_core import prompts
from prompts import prompt_manager
from utils import create_logger
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)


logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

def create_agent(
    instruction: str, 
    tools: Union[None, List[tool]],
    llm_model: str = config['AGENT_DEFAULT']['model'],
    ):

    # Define prompt template
    prompt = {
        # "history": lambda x: x["history"],
        "input": lambda x: x["input"],
        "agent_scratchpad": (lambda x: format_to_tool_messages(x["intermediate_steps"])),
        "dynamic_context": lambda x: x["dynamic_context"],
    } | prompts.ChatPromptTemplate.from_messages(
        [
            ("system", instruction.format(dynamic_context="{dynamic_context}")),
            # prompts.MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = reasoning_engines.LangchainAgent(
        prompt=prompt,
        model=llm_model,
        model_kwargs={"temperature": 0},
        tools=tools,
        agent_executor_kwargs={"return_intermediate_steps": False},
    )
    return agent

tools = [tool_collection.search_data_store, tool_collection.square]
instruction = prompt_manager.intro_prompt
coordinator_agent = create_agent(instruction=prompt_manager.intro_prompt, tools=tools)
search_cls_agent = create_agent(instruction=prompt_manager.second_prompt, tools=tools)
agent_dict = {
    "coordinator_agent": coordinator_agent,
    "search_cls_agent": search_cls_agent,
}
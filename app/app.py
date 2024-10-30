from agents import agent_manager
import configparser
import langchain
import tomllib
from tools import tool1

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

addition_toolkit = [
    tool1.add,
    tool1.multiply,
    tool1.square,
    tool1.search_datastore,
]

test_agent = agent_manager.AGENT(
    model=config['AGENT_DEFAULT']['model'],  # Required.
    tools=addition_toolkit,  # Optional.
    project=config['AGENT_DEFAULT']['project_id'],
    location=config['AGENT_DEFAULT']['location'],
)
test_agent.set_up()

test_agent.query(
    input='what is prompt engineering?'
)
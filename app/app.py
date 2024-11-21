import configparser
import tomllib
from tools import tool_collection 
from agents import agent_structure
from utils.utils import GraphNode
import prompts.prompt_manager as prompt_manager


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

my_model = config['AGENT_DEFAULT']['model']
my_project = config['AGENT_DEFAULT']['project_id']
my_location = config['AGENT_DEFAULT']['location']

addition_toolkit = [
    tool_collection.add,
]


d = GraphNode(name="controlAgent", agent=controlAgent)


r = d("What is 12+21")
print(r)

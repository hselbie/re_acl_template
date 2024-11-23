from agents import agent_structure 
from tools import tool_collection
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

my_model = config['AGENT_DEFAULT']['model']
my_project = config['AGENT_DEFAULT']['project_id']
my_location = config['AGENT_DEFAULT']['location']

planner_agent = agent_structure.PlannerAgent(
    model=my_model, 
    project=my_project, 
    location=my_location
    )
adder_agent = agent_structure.AdderAgent(
    model=my_model, 
    project=my_project, 
    location=my_location,
    tools=[tool_collection.add]
    )

factorator = agent_structure.FactAgent(
    model=my_model,
    project=my_project,
    location=my_location,
    # tools=[tool_collection.search]
)

checker_agent = agent_structure.CheckerAgent(
    model=my_model, 
    project=my_project, 
    location=my_location
    )

agents = {
            "planner": planner_agent,
            "adder": adder_agent,
            "factorator": factorator,
            "checker": checker_agent 
        }
     

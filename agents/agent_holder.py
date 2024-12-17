from agents import agent_structure 
from vertexai.preview.generative_models import ToolConfig
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
blood_agent = agent_structure.BloodAgent(
    model=my_model, 
    project=my_project, 
    location=my_location,
    tools=[tool_collection.search_blood_docs],
    # model_tool_kwargs={
    # "tool_config": {  # Specify the tool configuration here.
    #     "function_calling_config": {
    #         "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
    #         "allowed_function_names": ["search_blood_docs"],
    #         },
    #     },
    # },
    )
google_agent = agent_structure.GoogleProductAgent(
    model=my_model,
    project=my_project,
    location=my_location,
    tools=[tool_collection.subtract, tool_collection.next_agent]
    )
checker_agent = agent_structure.CheckerAgent(
    model=my_model, 
    project=my_project, 
    location=my_location
    )

agents = {
            "planner": planner_agent,
            "blood_donation": blood_agent,
            "google_product": google_agent,
            "checker": checker_agent 
        }
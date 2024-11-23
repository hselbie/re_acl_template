from agents import agent_structure, agent_holder
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

my_model = config['AGENT_DEFAULT']['model']
my_project = config['AGENT_DEFAULT']['project_id']
my_location = config['AGENT_DEFAULT']['location']



print(agent_structure.run_workflow(
    question='what city is the capital of france?',
    agents=agent_holder.agents))
from agents import agent_structure, agent_holder
import tomllib
from vertexai.preview import reasoning_engines
import vertexai

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

my_model = config['AGENT_DEFAULT']['model']
my_project = config['AGENT_DEFAULT']['project_id']
my_location = config['AGENT_DEFAULT']['location']
my_staging_bucket = f"gs://{config['AGENT_DEFAULT']['staging_bucket']}"

vertexai.init(
    project=my_project, 
    location=my_location, 
    staging_bucket=my_staging_bucket)
     
simple_agentic_flow = agent_structure.WorkflowManager(agents=agent_holder.agents)
simple_agentic_flow.set_up()

message = "how do i donate blood ?"

# remote_agent = reasoning_engines.ReasoningEngine.create(
#     agent_structure.WorkflowManager(agents=agent_holder.agents),
#     requirements=[
#         "google-cloud-aiplatform[langchain,reasoningengine]",
#         "cloudpickle==3.0.0",
#         "pydantic==2.7.4",
#         "langgraph",
#         "httpx",
#     ],
#     display_name="SimpleAgentFlow",
#     description="This is demo code",
#     extra_packages=[
#         "agents/",
#         "tools/", 
#         "utils/"],)

print(simple_agentic_flow.query(question=message, user='hugo'))
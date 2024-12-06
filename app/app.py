from agents import agent_structure, agent_holder
import tomllib
from vertexai.preview import reasoning_engines



with open("config.toml", "rb") as f:
    config = tomllib.load(f)

my_model = config['AGENT_DEFAULT']['model']
my_project = config['AGENT_DEFAULT']['project_id']
my_location = config['AGENT_DEFAULT']['location']

simple_agentic_flow = agent_structure.WorkflowManager(agents=agent_holder.agents)
simple_agentic_flow.set_up()

message = "What is the weather in Frances?"

remote_agent = reasoning_engines.ReasoningEngine.create(
    simple_agentic_flow(agents=agent_holder.agents),
    requirements=[
        "google-cloud-aiplatform[langchain,reasoningengine]",
        "cloudpickle==3.0.0",
        "pydantic==2.7.4",
        "langgraph",
        "httpx",
    ],
    display_name="Example MultiAgent Reasoning Engine with LangGraph",
    description="This is demo code",
    extra_packages=[],)

print(remote_agent.query(question=message, user='hugo'))
from vertexai.preview import reasoning_engines
import vertexai
from langchain_core import prompts
import requests 
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages

# Initialize Vertex AI (replace with your project/location)
vertexai.init(project="zinc-forge-302418", location="us-central1") # Don't init here - use the agent constructor

# Minimal Prompt
prompt = prompts.ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)
custom_prompt_template = {
    "user_input": lambda x: x["input"],
    # "history": lambda x: x["history"],
    "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
} | ChatPromptTemplate.from_messages([
    ("placeholder", "{history}"),
    ("user", "{user_input}"),
    ("placeholder", "{agent_scratchpad}"),
])

def add(x: int, y: int) -> int:
    """
    Calculate the sum of two integers.

    This function takes in two integer values, `x` and `y`, and returns their sum. 
    The result is computed by adding the two input values together.

    Parameters:
    x (int): The first integer to be added.
    y (int): The second integer to be added.

    Returns:
    int: The sum of the two input integers.

    Examples:
    ---------
    >>> add(3, 5)
    8

    >>> add(-2, 7)
    5

    >>> add(0, 0)
    0

    >>> add(-3, -6)
    -9
    """
    result = x + y
    return result

def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date.

    Uses the Frankfurter API (https://api.frankfurter.app/) to obtain
    exchange rate data.

    Args:
        currency_from: The base currency (3-letter currency code).
            Defaults to "USD" (US Dollar).
        currency_to: The target currency (3-letter currency code).
            Defaults to "EUR" (Euro).
        currency_date: The date for which to retrieve the exchange rate.
            Defaults to "latest" for the most recent exchange rate data.
            Can be specified in YYYY-MM-DD format for historical rates.

    Returns:
        dict: A dictionary containing the exchange rate information.
            Example: {"amount": 1.0, "base": "USD", "date": "2023-11-24",
                "rates": {"EUR": 0.95534}}
    """
    import requests
    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()

agent = reasoning_engines.LangchainAgent(
    prompt=custom_prompt_template,
    tools=[add, get_exchange_rate],
    model='gemini-pro',                # Required.
)

test = []
for i in range(0, 5):
    print(i)
    result = agent.query(
    input="What is 5+555?")
    test.append(result)
print(custom_prompt_template)
print(test)


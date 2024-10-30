import re
import uuid
from langchain_core.tools import StructuredTool
from langchain_google_community import VertexAISearchRetriever
from utils import create_logger
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings


logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

def search_datastore(query: str) -> str:
    """Tool to search a VAIS datastore using the langchain
    google community package for ease.

    Args:
        query (str): input query

    Returns:
        str: VAIS response 
    """
    retriever = VertexAISearchRetriever(
        project_id=config['AGENT_DEFAULT']['project_id'],
        data_store_id=config['TEST']['data_store_id'],
        location_id=config['TEST']['data_store_location'],
        engine_data_type=0,
        max_documents=10,
    )

    result = str(retriever.invoke(query))

    logger.info("search_data_store() returns:", result)

    return result

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def square(a: int) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

def create_tool(company: str) -> dict:
    """Create schema for a placeholder tool."""
    # Remove non-alphanumeric characters and replace spaces with underscores for the tool name
    formatted_company = re.sub(r"[^\w\s]", "", company).replace(" ", "_")

    def company_tool(year: int) -> str:
        # Placeholder function returning static revenue information for the company and year
        return f"{company} had revenues of $100 in {year}."

    return StructuredTool.from_function(
        company_tool,
        name=formatted_company,
        description=f"Information about {company}",
    )


# Abbreviated list of S&P 500 companies for demonstration
s_and_p_500_companies = [
    "3M",
    "A.O. Smith",
    "Abbott",
    "Accenture",
    "Advanced Micro Devices",
    "Yum! Brands",
    "Zebra Technologies",
    "Zimmer Biomet",
    "Zoetis",
]

# Create a tool for each company and store it in a registry with a unique UUID as the key
tool_registry = {
    str(uuid.uuid4()): create_tool(company) for company in s_and_p_500_companies
}

tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name},
    )
    for id, tool in tool_registry.items()
]

# Initialize the a specific Embeddings Model version
embedding_engine = VertexAIEmbeddings(model_name="text-embedding-004")

vector_store = InMemoryVectorStore(embedding=embedding_engine)
document_ids = vector_store.add_documents(tool_documents)
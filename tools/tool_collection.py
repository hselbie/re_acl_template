from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai.generative_models import GenerationConfig
from utils import create_logger
from langchain.tools import tool
import vertexai
from langchain_core.tools import tool
from typing import Optional

import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

@tool
def search_blood_docs(query: str, corpus:str) -> str:
    """
    If there is a query related to blood donation this function will perform RAG 
    (Retrieval-Augmented Generation) based search on blood-donation information corpus.

    Args:
        query (str): The search query or question to be processed
        corpus (str): The document corpus to search through

    Returns:
        str: The generated response from the Gemini model based on the RAG retrieval

    Notes:
        - Uses Vertex AI RAG store for document retrieval
        - Configures similarity search with top-k=3 and distance threshold=0.5
        - Utilizes the gemini-1.5-flash-001 model for response generation
        - Prints the response text before returning

    Example:
        >>> response = search_blood_docs("What are the symptoms of anemia?", blood_docs_corpus)
        >>> print(response)
    """
    rag_corpus = corpus 

    # Create a RAG retrieval tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=rag_corpus,  
                    )
                ],
                similarity_top_k=3,  # Optional
                vector_distance_threshold=0.5,  # Optional
            ),
        )
    )
    # Create a gemini-pro model instance
    rag_model = GenerativeModel(
        model_name="gemini-1.5-flash-001", tools=[rag_retrieval_tool]
    )

    # Generate response
    response = rag_model.generate_content(query)
    print(response.text)
    return response

@tool
def search_general_docs(query: str) -> str:
    """Search through general conversation documents. Use this for any non-Google/Alphabet related queries."""
    retriever = VertexAISearchRetriever(
        project_id="vertex-platform",
        location="us",
        search_engine_id="alphabet-pdfs_1733501864666",
        datastore_id="alphabet-pdfs_1733501891216",
        max_documents=5
    )
    results = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results])



@tool
def add(x: int, y: int) -> int:
    '''test function'''
    result = x+y
    return result

@tool
def next_agent():
    '''next agent function'''
    pass


def subtract(project: str):
    '''test function to return weather with a controlled generation'''
    my_project = config['AGENT_DEFAULT']['project_id']
    my_location = config['AGENT_DEFAULT']['location']

    vertexai.init(project=my_project, location=my_location)

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "forecast": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "Day": {"type": "STRING", "nullable": True},
                        "Forecast": {"type": "STRING", "nullable": True},
                        "Temperature": {"type": "INTEGER", "nullable": True},
                        "Humidity": {"type": "STRING", "nullable": True},
                        "Wind Speed": {"type": "INTEGER", "nullable": True},
                    },
                    "required": ["Day", "Temperature", "Forecast", "Wind Speed"],
                },
            }
        },
    }

    prompt = """
        The week ahead brings a mix of weather conditions.
        Sunday is expected to be sunny with a temperature of 77°F and a humidity level of 50%. Winds will be light at around 10 km/h.
        Monday will see partly cloudy skies with a slightly cooler temperature of 72°F and the winds will pick up slightly to around 15 km/h.
        Tuesday brings rain showers, with temperatures dropping to 64°F and humidity rising to 70%.
        Wednesday may see thunderstorms, with a temperature of 68°F.
        Thursday will be cloudy with a temperature of 66°F and moderate humidity at 60%.
        Friday returns to partly cloudy conditions, with a temperature of 73°F and the Winds will be light at 12 km/h.
        Finally, Saturday rounds off the week with sunny skies, a temperature of 80°F, and a humidity level of 40%. Winds will be gentle at 8 km/h.
    """

    model = GenerativeModel("gemini-1.5-pro-002")

    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json", response_schema=response_schema
        ),
    )

    print(response.text)
    return response.text
    # Example response:
    #  {"forecast": [{"Day": "Sunday", "Forecast": "Sunny", "Temperature": 77, "Humidity": "50%", "Wind Speed": 10},
    #     {"Day": "Monday", "Forecast": "Partly Cloudy", "Temperature": 72, "Wind Speed": 15},
    #     {"Day": "Tuesday", "Forecast": "Rain Showers", "Temperature": 64, "Humidity": "70%"},
    #     {"Day": "Wednesday", "Forecast": "Thunderstorms", "Temperature": 68},
    #     {"Day": "Thursday", "Forecast": "Cloudy", "Temperature": 66, "Humidity": "60%"},
    #     {"Day": "Friday", "Forecast": "Partly Cloudy", "Temperature": 73, "Wind Speed": 12},
    #     {"Day": "Saturday", "Forecast": "Sunny", "Temperature": 80, "Humidity": "40%", "Wind Speed": 8}]}

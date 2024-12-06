from vertexai.generative_models import GenerationConfig, GenerativeModel
from utils import create_logger
from langchain.tools import tool
import vertexai
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)


logger = create_logger.init_logger(__name__, testing_mode='DEBUG')

@tool
def add(x: int, y: int) -> int:
    '''test function'''
    result = x+y
    return result


def test_controlled_gen(project: str):
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

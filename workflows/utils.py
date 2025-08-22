import re
from typing import Literal, Annotated
import anthropic
import json
import os
from anthropic.types import Message
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
import requests

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")

anthropic_client = anthropic.Anthropic()


def claude(
    user_prompt: str = "",
    messages: list[any] = [],
    system_prompt="",
    model=MODEL_NAME,
    tools=[],
) -> str:
    """
    Calls Claude with the given user prompt, system prompt, and tools (if provided)
    Returns list of TextBlock or a ToolUseBlock
    """
    if not user_prompt and not messages:
        raise ValueError("Must provide at least one of user_prompt or messages")
    if not messages:
        messages = [{"role": "user", "content": user_prompt}]

    response: Message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        temperature=0.7,
        messages=messages,
        tools=tools if tools else anthropic.NOT_GIVEN,
    )

    return response


def simple_claude(
    user_prompt: str = "",
    messages: list[any] = [],
    system_prompt="",
    model=MODEL_NAME,
    tools=[],
) -> str:
    """
    Calls Claude with the given user prompt, system prompt, and tools (if provided)
    Returns a SINGLE TextBlock or ToolUseBlock
    """
    if not user_prompt and not messages:
        raise ValueError("Must provide at least one of user_prompt or messages")
    if not messages:
        messages = [{"role": "user", "content": user_prompt}]

    response: Message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        temperature=0.7,
        messages=messages,
        tools=tools if tools else anthropic.NOT_GIVEN,
    )

    print(f"Claude says: {response}")

    return response.content[0]


# define default schema for response
class Response(BaseModel):
    text: str


# util to call Claude with structured output!
# when we wrap Claude with instructor, Claude will automatically perform tool calls and return the response after making tool calls
def structured_claude(
    user_prompt: str = "",
    messages: list[dict] = [],
    response_model=Response,
    system_prompt="",
    model=MODEL_NAME,
    tools=[],
    parallel_tools=False,
) -> str:
    """
    Util to call Claude with structured output!
    Wraps Claude with the instructor library.
    Claude is capable of making multiple tools calls, but make sure the response_model is an Iterable
    """
    if not user_prompt and not messages:
        raise ValueError("Must provide at least one of user_prompt or messages")
    if not messages:
        messages = [{"role": "user", "content": user_prompt}]

    if parallel_tools:
        structured_client = instructor.from_anthropic(
            anthropic_client, mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS
        )
    else:
        structured_client = instructor.from_anthropic(
            anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS
        )
    response: Message = structured_client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        temperature=0.7,
        messages=messages,
        response_model=response_model,
        tools=tools if tools else None,
    )

    print(f"Claude says: {response}")
    return response


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts content between the opening and closing XML tags.
    Useful for parsing structured responses in XML format

    Args:
        text (str): the text containing XML tags
        tag (str): XML tags to extract content from

    Returns:
        str: content inside the XML tags
    """
    # by default, dot in regex matches any character except a newline
    # when re.DOTALL is enabled, the dot is modified to match all characters including new lines
    return re.search(f"<{tag}(.*?)</{tag}>", text, re.DOTALL)


class WebResult(BaseModel):
    title: str
    url: str
    is_source_local: bool
    is_source_both: bool
    description: str
    page_age: str | None = None
    profile: dict | None = None
    language: str
    family_friendly: bool
    type: Literal["search_result"]
    subtype: str
    is_live: bool
    deep_results: dict | None = None
    meta_url: dict
    thumbnail: dict | None = None
    age: str | None = None


class VideoResult(BaseModel):
    title: str
    url: str
    description: str
    fetched_content_timestamp: int | None = None
    video: dict
    type: Literal["video_result"]
    meta_url: dict


class BraveSearchResult(BaseModel):
    result: Annotated[WebResult | VideoResult, Field(discriminator="type")]


class Error(BaseModel):
    status_code: int
    content: str


def parse_brave_results(results: list[dict]) -> list[BraveSearchResult]:
    parsed_results = []
    for result in results:
        if result["type"] == "search_result":
            web_result = WebResult(**result)
            parsed_results.append(web_result)
        elif result["type"] == "video_result":
            video_result = VideoResult(**result)
            parsed_results.append(video_result)
        else:
            # TODO: handle other result types later
            continue

    return parsed_results


def brave_search(
    query: str, count: int = 10, result_filter: str = ""
) -> list[BraveSearchResult] | dict:
    """
    params:
        - query (required)
        - count (optional)
        - result_filter (optional): comma delimited string of result types to include in the search response.
            - leave blank to return all result types in the search response with available data
            - available result filter values: discussions, faq, infobox, news, query, summarizer, videos, web, locations
    """
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

    brave_search_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query, "count": count, "search_lang": "en", "country": "US"}
    if result_filter:
        params["result_filter"] = result_filter

    response = requests.get(brave_search_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        web_results = data.get("web", {}).get("results", [])
        video_results = data.get("videos", {}).get("results", [])
        locations_results = data.get("locations", {}).get("results", [])
        combined_results = web_results + video_results
        results = parse_brave_results(combined_results)
        return results
    else:
        error = Error(
            status_code=response.status_code, content=json.dumps(response.json())
        )
        return error


def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: str = "",
    type: int = 1,
    sort_by: int = 1,
):
    """
    Params:
        - departure_id: airport code or location kgmid (e.g. CDG for Paris Charles de Gaulle). Can also be a comma-delimited string of multiple ids
        - arrival_id: airport code or location kgmid
        - outbound_date: date in the format YYYY-MM-DD
        - return_date: Optional parameter to define the return date in the format YYYY-MM-DD. Required if the flight type = round trip
        - type: Optional parameter to define the type of the flight.
            - 1 = round trip
            - 2 = one way
            - 3 = multi-city
        - sort_by: Optional parameter to define the sorting order of the results.
            - 1 = Top flights (default)
            - 2 = Price
            - 3 = Departure time
            - 4 = Arrival time
            - 5 = Duration
            - 6 = Emissions
    """
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")

    serpapi_url = "https://serpapi.com/search.json?engine=google_flights"
    params = {
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": type,
        "sort_by": sort_by,
        "hl": "en",
        "gl": "us",
        "currency": "USD",
        "api_key": SERPAPI_KEY,
    }
    if return_date:
        params["return_date"] = return_date

    response = requests.get(serpapi_url, params=params)

    if response.status_code == 200:
        data = response.json()
        best_flights = data.get("best_flights", [])
        other_flights = data.get("other_flights", [])

        if best_flights:
            return best_flights
        else:
            return other_flights
    else:
        error = Error(
            status_code=response.status_code, content=json.dumps(response.json())
        )
        return error


def search_weather(
    latitude: float, longitude: float, forecast_days: int = 7, past_days: int = 0
):
    """
    Params:
        - latitude: required parameter
        - longitude: required parameter
        - forecast_days: optional parameter. How many days in the future the retrieved forecast should include. Maximum is 16.
        - past_days: optional parameter. How many days in the past to retrieve weather information for.
    """
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"

    latitude = 37.8716
    longitude = -122.2728
    forecast_days = 7
    past_days = 0
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "past_days": past_days,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "sunrise",
            "sunset",
            "precipitation_probability_max",
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_mean",
            "weather_code",
            "snowfall_sum",
            "showers_sum",
            "precipitation_hours",
        ],
        "timezone": "GMT",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }

    response = requests.get(open_meteo_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        error = Error(
            status_code=response.status_code, content=json.dumps(response.json())
        )
        return error


def get_url(url: str):
    """
    Params:
        - url: required parameter. The full url to fetch data from.
    """

    url = "https://example.com"
    response = requests.get(url)

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
        else:
            data = response.text
        return data
    else:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            error_content = json.dumps(response.json())
        else:
            error_content = response.text
        error = Error(status_code=response.status_code, content=error_content)
        return error


tools = [
    {
        "name": "brave_search",
        "description": "REST API to get search results from the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requred parameter. The query to search for. Query cannot be empty. Maximum of 400 characters",
                },
                "count": {
                    "type": "integer",
                    "description": "Optional parameter. The number of results to return. Maximum is 20. Defaults to 10",
                },
                "result_filter": {
                    "type": "string",
                    "description": """Optional parameter. A comma delimited string of result types to include in the search results.
- Leave blank to return all result types in the search response with available data
- Available result filter values: discussions, faq, infobox, news, query, summarizer, videos, web, locations
- Example: "web,videos" will return only web and video results.""",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_flights",
        "description": "Search for flight information (airline, price, duration, etc.) between two locations",
        "input_schema": {
            "type": "object",
            "properties": {
                "departure_id": {
                    "type": "string",
                    "description": "Required parameter. The airport code or location kgmid of the departure location (e.g. SFO or either JFK, LGA, or EWR for airports near NYC)",
                },
                "arrival_id": {
                    "type": "string",
                    "description": "Required parameter. The airport or location kgmid of the arrival location (e.g. SFO or either JFK, LGA, or EWR for airports near NYC)",
                },
                "outbound_date": {
                    "type": "string",
                    "description": "Required parameter. A date in the format YYYY-MM-DD (e.g. 2025-08-21)",
                },
                "return_date": {
                    "type": "string",
                    "description": "Optional parameter. A date in the format YYYY-MM-DD (e.g. 2025-08-21). Required if the flight type is round trip",
                },
                "type": {
                    "type": "integer",
                    "description": "Optional parameter to define the type of flight to search for. 1 = round trip, 2 = one way, 3 = multi-city",
                },
            },
            "required": ["departure_id, arrival_id, outbound_date"],
        },
    },
    {
        "name": "search_weather",
        "description": "Search for weather information (weather, temperature, sunrise/sunset, precipitation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Required parameter. The latitude of the location.",
                },
                "longitude": {
                    "type": "number",
                    "description": "Required parameter. The longitude of the location.",
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Optional parameter. How many days in the future the retrieved forecast should include. Maximum is 16.",
                },
                "past_days": {
                    "type": "integer",
                    "description": "Optional parameter. How many days in the past to retrieve weather information for.",
                },
            },
            "required": ["latitude, longitude"],
        },
    },
    {
        "name": "get_url",
        "description": "Get the webpage content from a url.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Required parameter. The full url of the webpage to get content from.",
                },
            },
            "required": ["url"],
        },
    },
]

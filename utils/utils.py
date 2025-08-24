"""Utility functions and classes for AI agent workflows.

This module provides functions for interacting with Claude AI models, web search APIs,
flight search, weather data, and URL content retrieval. It also includes Pydantic
models for structured data handling and tool definitions for AI agent interactions.

Main components:
- Claude AI interaction functions (claude, simple_claude, structured_claude)
- Web search via Brave Search API (brave_search)
- Flight search via SerpAPI (search_flights)
- Weather data via Open-Meteo API (search_weather)
- URL content retrieval (get_url)
- Pydantic models for data validation
- Tool definitions for AI agent workflows
"""

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
import tempfile
import fitz  # PyMuPDF

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")

anthropic_client = anthropic.Anthropic()


def claude(
    user_prompt: str = "",
    system_prompt="",
    messages: list[any] = [],
    model=MODEL_NAME,
    tools=[],
    max_tokens=1000
) -> str:
    """Calls Claude with the given user prompt, system prompt, and tools (if provided).

    Args:
        user_prompt: The user's input prompt. Defaults to empty string.
        system_prompt: System prompt to guide Claude's behavior. Defaults to empty string.
        messages: List of conversation messages. Defaults to empty list.
        model: Claude model to use. Defaults to MODEL_NAME environment variable.
        tools: List of tools available to Claude. Defaults to empty list.
        max_tokens: Maximum number of tokens to generate in the response. Defaults to 1000.

    Returns:
        Message: Claude's response containing TextBlock or ToolUseBlock.

    Raises:
        ValueError: If neither user_prompt nor messages are provided.
    """
    if not user_prompt and not messages:
        raise ValueError("Must provide at least one of user_prompt or messages")
    if not messages:
        messages = [{"role": "user", "content": user_prompt}]

    response: Message = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
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
    """Calls Claude with the given user prompt, system prompt, and tools (if provided).

    Args:
        user_prompt: The user's input prompt. Defaults to empty string.
        messages: List of conversation messages. Defaults to empty list.
        system_prompt: System prompt to guide Claude's behavior. Defaults to empty string.
        model: Claude model to use. Defaults to MODEL_NAME environment variable.
        tools: List of tools available to Claude. Defaults to empty list.

    Returns:
        str: A single TextBlock or ToolUseBlock content.

    Raises:
        ValueError: If neither user_prompt nor messages are provided.
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


class Response(BaseModel):
    """Default response schema for Claude structured output.

    Attributes:
        text: The text content of the response.
    """

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
    """Calls Claude with structured output using the instructor library.

    Wraps Claude with the instructor library to enable structured responses.
    Claude is capable of making multiple tool calls, but ensure the response_model
    is an Iterable if you expect multiple results.

    Args:
        user_prompt: The user's input prompt. Defaults to empty string.
        messages: List of conversation messages. Defaults to empty list.
        response_model: Pydantic model for structured output. Defaults to Response.
        system_prompt: System prompt to guide Claude's behavior. Defaults to empty string.
        model: Claude model to use. Defaults to MODEL_NAME environment variable.
        tools: List of tools available to Claude. Defaults to empty list.
        parallel_tools: Whether to use parallel tool execution. Defaults to False.

    Returns:
        Message: Claude's structured response matching the response_model.

    Raises:
        ValueError: If neither user_prompt nor messages are provided.
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
    """Extracts content between the opening and closing XML tags.

    Useful for parsing structured responses in XML format.

    Args:
        text: The text containing XML tags.
        tag: XML tag name to extract content from.

    Returns:
        str: Content inside the XML tags, or None if not found.
    """
    # by default, dot in regex matches any character except a newline
    # when re.DOTALL is enabled, the dot is modified to match all characters including new lines
    return re.search(f"<{tag}(.*?)</{tag}>", text, re.DOTALL)


class WebResult(BaseModel):
    """Schema for web search results from Brave Search API.

    Attributes:
        title: Title of the webpage.
        url: URL of the webpage.
        is_source_local: Whether the source is local.
        is_source_both: Whether the source is both local and non-local.
        description: Description or snippet of the webpage content.
        page_age: Age of the page content. Optional.
        profile: Profile information. Optional.
        language: Language of the webpage.
        family_friendly: Whether the content is family-friendly.
        type: Type identifier, always "search_result".
        subtype: Subtype of the search result.
        is_live: Whether the content is live/real-time.
        deep_results: Additional deep search results. Optional.
        meta_url: Metadata about the URL.
        thumbnail: Thumbnail image information. Optional.
        age: Age classification. Optional.
    """

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
    """Schema for video search results from Brave Search API.

    Attributes:
        title: Title of the video.
        url: URL of the video.
        description: Description of the video content.
        fetched_content_timestamp: Timestamp when content was fetched. Optional.
        video: Video-specific metadata and information.
        type: Type identifier, always "video_result".
        meta_url: Metadata about the URL.
    """

    title: str
    url: str
    description: str
    fetched_content_timestamp: int | None = None
    video: dict
    type: Literal["video_result"]
    meta_url: dict


class BraveSearchResult(BaseModel):
    """Union type for Brave Search results, discriminated by result type.

    Attributes:
        result: Either a WebResult or VideoResult, discriminated by the 'type' field.
    """

    result: Annotated[WebResult | VideoResult, Field(discriminator="type")]


class Error(BaseModel):
    """Schema for error responses from API calls.

    Attributes:
        status_code: HTTP status code of the error.
        content: Error message or content details.
    """

    status_code: int
    content: str


def parse_brave_results(results: list[dict]) -> list[BraveSearchResult]:
    """Parses raw Brave Search API results into structured Pydantic models.

    Args:
        results: List of raw result dictionaries from Brave Search API.

    Returns:
        list[BraveSearchResult]: List of parsed WebResult and VideoResult objects.

    Note:
        Currently handles 'search_result' and 'video_result' types.
        Other result types are skipped for future implementation.
    """
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
    """Searches the web using Brave Search API.

    Args:
        query: The search query string. Required.
        count: Number of results to return. Maximum is 20. Defaults to 10.
        result_filter: Comma-delimited string of result types to include.
            Leave blank to return all result types. Available values:
            discussions, faq, infobox, news, query, summarizer, videos, web, locations.
            Defaults to empty string.

    Returns:
        list[BraveSearchResult] | dict: List of search results or error dict.

    Note:
        Requires BRAVE_API_KEY environment variable to be set.
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
    flight_type: int = 1,
    sort_by: int = 1,
):
    """Searches for flight information using Google Flights via SerpAPI.

    Args:
        departure_id: Airport code or location kgmid (e.g., CDG for Paris Charles de Gaulle).
            Can also be a comma-delimited string of multiple IDs.
        arrival_id: Airport code or location kgmid of the arrival location.
        outbound_date: Departure date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format. Required if flight type is round trip.
            Defaults to empty string.
        flight_type: Flight type. 1 = round trip, 2 = one way, 3 = multi-city. Defaults to 1.
        sort_by: Sorting order for results. 1 = Top flights, 2 = Price, 3 = Departure time,
            4 = Arrival time, 5 = Duration, 6 = Emissions. Defaults to 1.

    Returns:
        list | dict: List of flight results or error dict.

    Note:
        Requires SERPAPI_KEY environment variable to be set.
    """
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")

    serpapi_url = "https://serpapi.com/search.json?engine=google_flights"
    params = {
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": flight_type,
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
    """Retrieves weather information for a specific location using Open-Meteo API.

    Args:
        latitude: Latitude coordinate of the location. Required.
        longitude: Longitude coordinate of the location. Required.
        forecast_days: Number of days in the future for forecast. Maximum is 16. Defaults to 7.
        past_days: Number of days in the past for historical weather. Defaults to 0.

    Returns:
        dict: Weather data including temperature, precipitation, sunrise/sunset times,
            or error dict if request fails.

    Note:
        Currently hardcoded to San Francisco coordinates (37.8716, -122.2728).
        Returns daily weather data in Fahrenheit, miles per hour, and inches.
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
    """Fetches content from a URL and returns the response data.

    Args:
        url: The full URL to fetch data from. Required.

    Returns:
        dict | str: JSON data if content-type is application/json, otherwise text content,
            or error dict if request fails.
    """
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


# List of tools available for Claude to use
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


def process_tool_call(tool_name: str, params: dict):
    print("=" * 50)
    print(f"Tool call using tool {tool_name}")
    print(f"Tool call params {params}")

    if tool_name == "brave_search":
        res = brave_search(**params)
    elif tool_name == "search_flights":
        res = search_flights(**params)
    elif tool_name == "search_weather":
        res = search_weather(**params)
    elif tool_name == "get_url":
        res = get_url(**params)
    else:
        res = f"The {tool_name} tool isn't currently defined"
    print(f"Tool call result: {res}")
    print("=" * 50)
    return res


def get_pdf_text(pdf_url: str) -> str | Error:
    response = requests.get(pdf_url)
    if response.status_code == 200:
        tempdir = tempfile.TemporaryDirectory()
        print(f"Temporary directory created at: {tempdir.name}")

        try:
            temp_pdf_path = os.path.join(tempdir.name, "temp.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(response.content)
            print(f"PDF downloaded successfully to {temp_pdf_path}")
            document = fitz.open(temp_pdf_path)
            pdf_text = []
            for page_index, page in enumerate(document):
                text = page.get_text("text")
                pdf_text.append(text)
            res = "\n".join(pdf_text)
            return res
        except Exception as e:
            print(f"Error extracting text from pdf: {e}")

        tempdir.cleanup()
        print("Temporary directory cleaned up")

    else:
        print(
            f"Status code {response.status_code} when getting pdf text. Error message: {response.json()}"
        )
        return Error(
            status_code=response.status_code, content=json.dumps(response.json())
        )

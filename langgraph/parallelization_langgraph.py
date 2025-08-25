import json
import sys
import os
from typing import TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Add the project root to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import brave_search, search_flights, search_weather, get_url

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")
model = init_chat_model(MODEL_NAME, model_provider="anthropic")

current_date = datetime.now().strftime("%Y-%m-%d")

int_to_day = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday", 
    6: "Saturday",
    7: "Sunday"
}

current_day_of_the_week = int_to_day[datetime.now().isoweekday()]

class State(TypedDict):
    user_input: str
    weather_result: str
    flight_result: str
    itinerary_result: str
    summary: str
    messages: Annotated[list, add_messages] = []


def process_tool_call(tool_name: str, params: dict):
    print("=" * 50)
    print(f"Tool call using tool {tool_name}")
    print(f"Tool call params {params}")

    if tool_name == "weather_tool":
        res = search_weather(**params)
    elif tool_name == "flights_tool":
        res = search_flights(**params)
    elif tool_name == "brave_search_tool":
        res = brave_search(**params)
    elif tool_name == "get_url_tool":
        res = get_url(**params)
    else:
        res = f"The {tool_name} tool isn't currently defined"
    if "</html>" not in res:
        print(f"Tool call result: {res}")
    print("=" * 50)
    return res


# Define all the tools
@tool(parse_docstring=True)
def weather_tool(
    latitude: float, longitude: float, forecast_days: int = 7, past_days: int = 0
):
    """Search for weather information (weather, temperature, sunrise/sunset, precipitation).

    Args:
        latitude
        longitude
        forecast_days: optional parameter How many days in the future the retrieved forecast should include. Maximum is 16.
        past_days: optional parameter. How many days in the past to retrieve weather information for.
    """
    return search_weather(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days,
        past_days=past_days,
    )


@tool(parse_docstring=True)
def flights_tool(
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
    """
    return search_flights(
        departure_id=departure_id,
        arrival_id=arrival_id,
        outbound_date=outbound_date,
        return_date=return_date,
        flight_type=flight_type,
        sort_by=sort_by,
    )


@tool(parse_docstring=True)
def brave_search_tool(query: str, count: int = 10, result_filter: str = ""):
    """Searches the web using Brave Search API.

    Args:
        query: The search query string. Required.
        count: Number of results to return. Maximum is 20. Defaults to 10.
        result_filter: Comma-delimited string of result types to include.
            Leave blank to return all result types.
            Available values are discussions, faq, infobox, news, query, summarizer, videos, web, locations.
            Defaults to empty string.

    Returns:
        list[BraveSearchResult] | dict: List of search results or error dict.
    """
    return brave_search(query=query, count=count, result_filter=result_filter)


@tool(parse_docstring=True)
def get_url_tool(url: str):
    """Fetches content from a URL and returns the response data.

    Args:
        url: The full URL to fetch data from. Required.

    Returns:
        dict | str: JSON data if content-type is application/json, otherwise text content,
            or error dict if request fails.
    """
    return get_url(url=url)


def flight_searcher(state: State) -> State:
    system_prompt = f"""
You are a helpful travel assistant.
Given a user input, find the best flight options for the user.
If the user doesn't specify a starting location, you can assume they are based in San Francisco.
If the user doesn't specify a date, use today's date. Or find the best flights over the next few days.
You can also try searching for narby airports.

Return information for the best flight option you can find based on your research.

For your information, today's date is {current_day_of_the_week}, {current_date}.
"""
    system_message = SystemMessage(content=system_prompt)
    user_input = state["user_input"]
    user_message = HumanMessage(content=user_input)
    messages = [system_message, user_message]

    model_with_tools = model.bind_tools([flights_tool])
    response = model_with_tools.invoke(messages)

    print("=" * 50)
    print("Response from the flight searcher:", response)
    print("=" * 50)

    messages.append(response)

    while hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            tool_result = process_tool_call(tool_name, tool_call_args)

            serialized_result = str(tool_result)

            tool_message = ToolMessage(
                content=json.dumps(serialized_result),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            messages.append(tool_message)

        response = model_with_tools.invoke(messages)
        messages.append(response)
    else:
        # if model decides not to make any more tool calls, append final response to the state
        if isinstance(response.content, list):
            flight_result = response.content[0].text
        else:
            flight_result = response.content
        # avoid re-adding multiple system messages to state
        return {"messages": messages[2:], "flight_result": flight_result}


def weather_checker(state: State) -> State:
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful meterologist.
Given a user input, find relevant weather information for the user.

For your information, today's date is {current_day_of_the_week}, {current_date}.
"""
    system_message = SystemMessage(content=system_prompt)
    user_input = state["user_input"]
    user_message = HumanMessage(content=user_input)
    messages = [system_message, user_message]

    model_with_tools = model.bind_tools([weather_tool])
    response = model_with_tools.invoke(messages)

    print("=" * 50)
    print("Response from the weather checker:", response)
    print("=" * 50)

    messages.append(response)

    while hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            tool_result = process_tool_call(tool_name, tool_call_args)

            serialized_result = str(tool_result)

            tool_message = ToolMessage(
                content=json.dumps(serialized_result),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            messages.append(tool_message)

        response = model_with_tools.invoke(messages)
        messages.append(response)
    else:
        # if model decides not to make any more tool calls, append final response to the state
        if isinstance(response.content, list):
            weather_result = response.content[0].text
        else:
            weather_result = response.content
        # avoid re-adding multiple system messages to state
        return {"messages": messages[2:], "weather_result": weather_result}


def itinerary_writer(state: State) -> State:
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful travel planner.
Given a user input, plan out a fun and interesting itinerary for the user.
For example, the itinerary may include points of interest, activities, local customs, and local foods.
Use the tools to research more when applicable. You can also get webpage content directly from a url using the tools.

For your information, today's date is {current_day_of_the_week}, {current_date}.
"""
    system_message = SystemMessage(content=system_prompt)
    user_input = state["user_input"]
    user_message = HumanMessage(content=user_input)
    messages = [system_message, user_message]

    model_with_tools = model.bind_tools([brave_search_tool, get_url_tool])
    response = model_with_tools.invoke(messages)

    print("=" * 50)
    print("Response from the itinerary writer:", response)
    print("=" * 50)

    messages.append(response)

    while hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            tool_result = process_tool_call(tool_name, tool_call_args)

            serialized_result = str(tool_result)

            tool_message = ToolMessage(
                content=json.dumps(serialized_result),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            messages.append(tool_message)

        response = model_with_tools.invoke(messages)
        messages.append(response)
    else:
        # if model decides not to make any more tool calls, append final response to the state
        if isinstance(response.content, list):
            itinerary_result = response.content[0].text
        else:
            itinerary_result = response.content
        # avoid re-adding multiple system messages to state
        return {"messages": messages[2:], "itinerary_result": itinerary_result}


def summarizer(state: State):
    system_prompt = """
You are a helpful summarizer.

Some other helpful AI assistants helped to compile flight information, weather information, and an itinerary.
Summarize all the information about flights, weather, and a proposed fun intinerary.
Note that the user will not be given the flight information, weather information, and itinerary you receive.
Therefore, you should include as much information about the intinerary as you can in your summary.
"""
    system_message = SystemMessage(content=system_prompt)
    user_input = state["user_input"]
    user_message = HumanMessage(content=user_input)
    messages = [system_message, user_message]
    flight_result = state.get("flight_result", "")
    weather_result = state.get("weather_result", "")
    itinerary_result = state.get("itinerary_result", "")

    messages = [
        HumanMessage(content=f"Flight information: {flight_result}"),
        HumanMessage(content=f"Weather information: {weather_result}"),
        HumanMessage(content=f"Itinerary: {itinerary_result}"),
    ]
    response = model.invoke(messages)

    print("=" * 50)
    print("Response from the summarizer:", response)
    print("=" * 50)

    return {"messages": response, "summary": response.content}


def main():
    # user_input = "I want to visit Paris next weekend. What are some upcoming exhibits at the Musee de l'Orangerie?"
    user_input = input("Where in the world do you want to explore? ")

    graph_builder = StateGraph(State)

    graph_builder.add_node("flight_searcher_node", flight_searcher)
    graph_builder.add_node("weather_checker_node", weather_checker)
    graph_builder.add_node("itinerary_writer_node", itinerary_writer)
    graph_builder.add_node("summary_node", summarizer)

    graph_builder.add_edge(START, "flight_searcher_node")
    graph_builder.add_edge(START, "weather_checker_node")
    graph_builder.add_edge(START, "itinerary_writer_node")

    graph_builder.add_edge("flight_searcher_node", "summary_node")
    graph_builder.add_edge("weather_checker_node", "summary_node")
    graph_builder.add_edge("itinerary_writer_node", "summary_node")

    graph_builder.add_edge("summary_node", END)

    graph = graph_builder.compile()

    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     pass

    final_state = graph.invoke({"user_input": user_input})
    for message in final_state["messages"]:
        print(f"Role: {type(message).__name__}")
        if type(message).__name__ != "ToolMessage":
            print(f"Message: {message.content}\n") if type(
                message
            ).__name__ != "ToolMessage" else print("\n")


if __name__ == "__main__":
    main()

import json
import sys
import os
from typing import TypedDict, Optional, Literal, Annotated
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Add the project root to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import brave_search, search_flights, search_weather, get_url

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")
model = init_chat_model(MODEL_NAME, model_provider="anthropic")


class State(TypedDict):
    user_input: str
    summary: str
    messages: Annotated[list, add_messages] = []


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
        sort_by=sort_by
    )


@tool(parse_docstring=True)
def brave_search_tool(
    query: str, count: int = 10, result_filter: str = ""
):
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
    return brave_search(
        query=query,
        count=count,
        result_filter=result_filter
    )


class ToolNode:
    """A node that runs the tools requested in the latest AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_map = {
            tool.name: tool for tool in tools
        }  # maps tool names to the tool

    def __call__(self, state: State):
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages in the state")

        message = messages[-1]

        outputs = []
        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            tool = self.tools_map[tool_name]
            tool_result = tool.invoke(tool_call_args)
            print("=" * 50)
            print(f"Calling tool with name {tool_name} and args {tool_call_args}")
            print(f"Tool call result: {tool_result}")
            print("=" * 50)

            serialized_result = str(tool_result)
            
            outputs.append(
                ToolMessage(
                    content=json.dumps(serialized_result),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
        return {"messages": outputs}


def travel_agent(state: State) -> State:
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful travel assistant.

Given a user input, help the user plan a trip.

Your trip summary should include the following information:
* Flight details
* Weather forecast
* A fun itinerary!

Make sure to use tools to make your response as detailed and up-to-date as possible

If the user doesn't specify a starting location, you can assume they are based in San Francisco.
If the user doesn't specify a date, use today's date. Or find the best flights over the next few days.

For your information, today's date is {current_date}
"""
    system_message = SystemMessage(content=system_prompt)
    if not state["messages"] or len(state["messages"]) == 0:
        user_input = state["user_input"]
        user_message = HumanMessage(content=user_input)
        messages = [system_message, user_message]
    else:
        messages = state["messages"]

    model_with_tools = model.bind_tools([flights_tool, weather_tool, brave_search_tool])
    response = model_with_tools.invoke(messages)

    print("=" * 50)
    print("Response from the travel agent:", response)
    print("=" * 50)

    messages.append(response)

    # if model decides not to make any more tool calls, append final response to the state
    if not hasattr(response, "tool_calls") or len(response.tool_calls) == 0:
        if isinstance(response.content, list):
            summary = response.content[0].text
        else:
            summary = response.content
        return {"messages": messages, "summary": summary}
    else:
        return {"messages": messages}


def route_tools(state: State):
    """
    Used in conditional_edges to route to the ToolNode if the last message
    in the state contains a tool call. Otherwise, routes to the end
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in the route_tools edge: {state}")
    
    # print("=" * 50)
    # print(f"ROUTING MESSAGE: {ai_message}")
    # print("=" * 50)

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tool_node"
    return END


def main():
    user_input = input("Where in the world do you want to explore? ")
    tools = [flights_tool, weather_tool, brave_search_tool]
    # for t in tools:
    #     print(f"Tool name: {t.name}")
    #     print(f"Tool description: {t.description}")
    #     print(f"Tool args: {t.args}")
    
    tool_node = ToolNode(tools)

    graph_builder = StateGraph(State)

    graph_builder.add_node("travel_agent_node", travel_agent)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "travel_agent_node")

    graph_builder.add_conditional_edges(
        "travel_agent_node", route_tools, {"tool_node": "tool_node", END: END}
    )
    # Any time a tool is called, we return to the agent to decide the next step
    graph_builder.add_edge("tool_node", "travel_agent_node")

    graph = graph_builder.compile()
    
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     pass

    final_state = graph.invoke({"user_input": user_input})

    for message in final_state["messages"]:
        print(f"Role: {type(message).__name__}")
        print(f"Message: {message.content}\n")


if __name__ == "__main__":
    main()

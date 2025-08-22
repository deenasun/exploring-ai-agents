from datetime import datetime
from utils import claude, search_flights, brave_search, search_weather, get_url, tools
import asyncio
from concurrent.futures import ThreadPoolExecutor


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


def tool_call_cycle_example(user_input: str):
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = f"""
You are a helpful web searching assistant.
Try to answer the user's query to the best of your abilities.
You can use tool calls if needed to search up more information.
Use the most applicable tool for the information you need.

For your information, today's date is {current_date}
"""

    user_input = "What is the weather like in Austin for the next 2 days?"

    messages = [{"role": "user", "content": user_input}]

    response = claude(messages=messages, system_prompt=system_prompt, tools=tools)

    print("Initial Response:")
    print(f"Message: {response}")
    print(f"Stop Reason: {response.stop_reason}")
    print(f"Content: {response.content}")

    tool_call_count = 0
    while tool_call_count < 5 and response.stop_reason == "tool_use":
        for block in response.content:
            messages.append({"role": "assistant", "content": [block]})

            if block.type == "tool_use":
                tool_call_count += 1
                tool_use_id = block.id
                tool_result = process_tool_call(
                    tool_name=block.name, params=block.input
                )

                print("TOOL USE RESULT", tool_result)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(tool_result),
                            }
                        ],
                    }
                )

        # Get next response if we haven't hit the limit
        if tool_call_count < 5:
            response = claude(
                messages=messages, system_prompt=system_prompt, tools=tools
            )
    else:
        final_response = response
        messages.append(final_response)
        print("=" * 50)
        print(f"Final response: {messages[-1].content[0].text}")
        print("=" * 50)


def flight_searcher(user_input: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful travel assistant.
Given a user input, find the best flight options for the user.
If the user doesn't specify a starting location, you can assume they are based in San Francisco.
If the user doesn't specify a date, use today's date. Or find the best flights over the next few days.
You can also try searching for narby airports.

Return a information for the best flight option you can find based on your research.
å
For your information, today's date is {current_date}
"""

    messages = [{"role": "user", "content": user_input}]

    flight_search_tools = [
        {
            "name": "search_flights",
            "description": "Search for flight information (airline, price, duration, etc.) between two locations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "departure_id": {
                        "type": "string",
                        "description": "Required parameter. The airport code of the airport nearest to the departure location (e.g. SFO or either JFK, LGA, or EWR for airports near NYC)",
                    },
                    "arrival_id": {
                        "type": "string",
                        "description": "Required parameter. The airport code of the airport nearest to the arrival location (e.g. SFO or either JFK, LGA, or EWR for airports near NYC)",
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
    ]

    response = claude(
        messages=messages, system_prompt=system_prompt, tools=flight_search_tools
    )

    print("=" * 50)
    print("Initial response from the flight searcher:")
    print("Initial Response:")
    print(f"Message: {response}")
    print(f"Stop Reason: {response.stop_reason}")
    print(f"Content: {response.content}")
    print("=" * 50)

    tool_call_count = 0
    while tool_call_count < 5 and response.stop_reason == "tool_use":
        for block in response.content:
            messages.append({"role": "assistant", "content": [block]})

            if block.type == "tool_use":
                tool_call_count += 1
                tool_use_id = block.id
                tool_result = process_tool_call(
                    tool_name=block.name, params=block.input
                )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(tool_result),
                            }
                        ],
                    }
                )

        # Get next response if we haven't hit the limit
        if tool_call_count < 5:
            response = claude(
                messages=messages, system_prompt=system_prompt, tools=tools
            )
    else:
        final_response = response
        messages.append(final_response)
        print("=" * 50)
        print(
            f"Final response from the flight searcher: {messages[-1].content[0].text}"
        )
        print("=" * 50)
        return messages[-1].content[0].text


def weather_checker(user_input: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful meterologist.
Given a user input, find relevant weather information for the user.

For your information, today's date is {current_date}
"""

    messages = [{"role": "user", "content": user_input}]

    weather_tools = [
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
    ]

    response = claude(
        messages=messages, system_prompt=system_prompt, tools=weather_tools
    )

    print("=" * 50)
    print("Initial response from the weather checker:")
    print("Initial Response:")
    print(f"Message: {response}")
    print(f"Stop Reason: {response.stop_reason}")
    print(f"Content: {response.content}")
    print("=" * 50)

    tool_call_count = 0
    while tool_call_count < 5 and response.stop_reason == "tool_use":
        for block in response.content:
            messages.append({"role": "assistant", "content": [block]})

            if block.type == "tool_use":
                tool_call_count += 1
                tool_use_id = block.id
                tool_result = process_tool_call(
                    tool_name=block.name, params=block.input
                )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(tool_result),
                            }
                        ],
                    }
                )

        # Get next response if we haven't hit the limit
        if tool_call_count < 5:
            response = claude(
                messages=messages, system_prompt=system_prompt, tools=tools
            )
    else:
        final_response = response
        messages.append(final_response)
        print("=" * 50)
        print(
            f"Final response from the weather checker: {messages[-1].content[0].text}"
        )
        print("=" * 50)
        return messages[-1].content[0].text


def itinerary_writer(user_input: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
You are a helpful travel planner.
Given a user input, plan out a fun and interesting itinerary for the user.
For example, the itinerary may include points of interest, activities, local customs, and local foods.
Use the tools to research more when applicable. You can also get webpage content directly from a url using the tools.

For your information, today's date is {current_date}
"""

    messages = [{"role": "user", "content": user_input}]

    itinerary_tools = [
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

    response = claude(
        messages=messages, system_prompt=system_prompt, tools=itinerary_tools
    )

    print("=" * 50)
    print("Initial response from the itinerary writer:")
    print("Initial Response:")
    print(f"Message: {response}")
    print(f"Stop Reason: {response.stop_reason}")
    print(f"Content: {response.content}")
    print("=" * 50)

    tool_call_count = 0

    while tool_call_count < 5 and response.stop_reason == "tool_use":
        for block in response.content:
            messages.append({"role": "assistant", "content": [block]})

            if block.type == "tool_use":
                tool_call_count += 1
                tool_use_id = block.id
                tool_result = process_tool_call(
                    tool_name=block.name, params=block.input
                )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(tool_result),
                            }
                        ],
                    }
                )

        # Get next response if we haven't hit the limit
        if tool_call_count < 5:
            response = claude(
                messages=messages, system_prompt=system_prompt, tools=tools
            )
    else:
        final_response = response
        messages.append(final_response)
        print("=" * 50)
        print(
            f"Final response from the itinerary writer: {messages[-1].content[0].text}"
        )
        print("=" * 50)
        return messages[-1].content[0].text


async def parallelization(user_input: str) -> None:
    user_input = "I want to visit Austin!"
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Need to wrap the synchronous functions in a coroutine
        # that can run in a thread pool using run_in_executor
        tasks = [
            loop.run_in_executor(executor, flight_searcher, user_input),
            loop.run_in_executor(executor, weather_checker, user_input),
            loop.run_in_executor(executor, itinerary_writer, user_input),
        ]

        results = await asyncio.gather(*tasks)
        flight_result, weather_result, itinerary_result = results

    # Or, without asyncio
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     # Submit all tasks
    #     flight_future = executor.submit(flight_searcher, user_input)
    #     weather_future = executor.submit(weather_checker, user_input)
    #     itinerary_future = executor.submit(itinerary_writer, user_input)

    #     # Wait for all to complete
    #     flight_result = flight_future.result()
    #     weather_result = weather_future.result()
    #     itinerary_result = itinerary_future.result()

    system_prompt = """
You are a helpful summarizer.
Given a user input, summarize all the information about flights, weather, and a proposed fun intinerary.
Note that the user will not be given the flight information, weather information, and itinerary you receive.
Therefore, you should include as much information about the intinerary as you can in your summary.
"""
    messages = [
        {"role": "assistant", "content": f"Flight information: {flight_result}"},
        {"role": "assistant", "content": f"Weather information: {weather_result}"},
        {"role": "assistant", "content": f"Itinerary: {itinerary_result}"},
        {"role": "user", "content": user_input},
    ]

    response = claude(messages=messages, system_prompt=system_prompt)

    print("Final Response:")
    print(f"Summary: {response.content[0].text}")

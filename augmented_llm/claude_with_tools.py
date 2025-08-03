import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")

anthropic_client = anthropic.Anthropic()

# response without structured output
message = anthropic_client.messages.create(
    model=MODEL_NAME,
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond in short poems.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why is the sky blue?"}],
        }
    ],
)

print(message)
print(message.content[0].text)


# define functions to use as tools
def add(a, b):
    """
    Adds two numbers
    """
    return a + b


def subtract(a, b):
    """
    Subtracts the second argument from the first argument
    """
    return a - b


def multiply(a, b):
    """
    Multiplies two numbers
    """
    return a * b


def divide(a, b):
    """
    Divides the first argument by the second argument
    """
    return a / b


tools = [
    {
        "name": "add",
        "description": "Adds two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "addend"},
                "b": {"type": "number", "description": "addend"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "subtract",
        "description": "Subtracts b from a",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "a"},
                "b": {"type": "number", "description": "number to subtract from a"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiplies two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "factor"},
                "b": {"type": "number", "description": "factor"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "divide",
        "description": "Divides a by b",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "dividend"},
                "b": {"type": "number", "description": "divisor"},
            },
            "required": ["a", "b"],
        },
    },
]


def process_tool_call(tool_name, tool_input):
    match tool_name:
        case "add":
            return add(tool_input["a"], tool_input["b"])
        case "subtract":
            return subtract(tool_input["a"], tool_input["b"])
        case "multiply":
            return multiply(tool_input["a"], tool_input["b"])
        case "divide":
            return divide(tool_input["a"], tool_input["b"])
        case _:  # default
            return f"Invalid tool call. Tool with name {tool_name} doesn't exist."


def chat_with_claude(user_message: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        }
    ]

    message = anthropic_client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        temperature=1,
        system="You are a genius mathematician! Use your tools to solve these math problems.",
        messages=messages,
        tools=tools,
    )

    print("Initial Response:")
    print(f"Message: {message}")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content: {message.content}")

    # # serializes response into a python dictionary
    # print(message.model_dump())

    # loop while the agent needs to keep using tools
    while message.stop_reason == "tool_use":
        # this gets the first element in the list that satisfies the condition
        # tool_use = next(block for block in message.content if block.type == "tool_use")
        # tool_name = tool_use.name
        # tool_input = tool_use.input

        for block in message.content:
            if block.type != "tool_use":
                continue

            messages.append({"role": "assistant", "content": [block]})

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            print(f"\nTool Used: {tool_name}")
            print(f"Tool Input: {tool_input}")

            tool_result = process_tool_call(tool_name, tool_input)

            print(f"Tool Result: {tool_result}")

            # Claude documentation recommends using user for tool results
            # The content will contain info that this was a tool_result
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

        message = anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=messages,
            tools=tools,
        )

    else:  # else executes after the while loop finishes
        response = message

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    print(response.content)
    print(f"\nFinal Response: {final_response}")

    return final_response


chat_with_claude("Calculate ((1 + 2) * 15) / 3 - 18")

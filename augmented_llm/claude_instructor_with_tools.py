from typing import Literal, Optional
import anthropic
import instructor
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import uuid

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")

anthropic_client = anthropic.Anthropic()

# wrap anthropic client with instructor to support structured output
client = instructor.from_anthropic(
    anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS,
)

# can also directly create a client and specify the model like this:
# client = instructor.from_provider(f"anthropic/{os.getenv("CLAUDE_MODEL")}"")


# Define the response format for structured output
class Response(BaseModel):
    text: str


# response with structured output
message = client.messages.create(
    model=os.getenv("CLAUDE_MODEL"),
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond in short poems.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why is the sky blue?"}],
        }
    ],
    response_model=Response,
    max_retries=2
)

print(message)
print(message.text)


# define structured output for possible tool call
class ToolCallOrAnswer(BaseModel):
    needs_tool: bool = Field(description="True if a tool call is needed, False if ready to provide final answer")
    tool_name: Optional[
        Literal["add", "subtract", "multiply", "divide"]
    ] = Field(None,
        description="Name of the tool to call, if needed. Can be either add, "
        "subtract, multiply, or divide."
    )
    tool_input: Optional[dict] = Field(None,
        description="Argument for the tool call, if needed. Provide it as a "
        "Python dictionary with the keys a and b."
    )
    final_answer: Optional[str] = Field(None, description="Final answer if no more tools are needed")


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
        "description": "Adds two numbers, a and b",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "parameter a"},
                "b": {"type": "number", "description": "parameter b"},
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
                "a": {"type": "number", "description": "parameter a"},
                "b": {"type": "number", "description": "parameter b"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiplies two numbers, a and b",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "parameter a"},
                "b": {"type": "number", "description": "parameter b"},
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
                "a": {"type": "number", "description": "parameter a"},
                "b": {"type": "number", "description": "parameter b"},
            },
            "required": ["a", "b"],
        },
    },
]


def process_tool_call(tool_name, args):
    match tool_name:
        case "add":
            try:
                return add(**args)
            except Exception as e:
                return f"Error: {e}. Please try again."
        case "subtract":
            try:
                return subtract(**args)
            except Exception as e:
                return f"Error: {e}. Please try again."
        case "multiply":
            try:
                return multiply(**args)
            except Exception as e:
                return f"Error: {e}. Please try again."
        case "divide":
            try:
                return divide(**args)
            except Exception as e:
                return f"Error: {e}. Please try again."
        case _:  # default
            return f"Invalid tool call. Tool with name {tool_name} doesn't exist."


def chat_with_claude(user_message: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        }
    ]

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        temperature=1,
        system="You are a genius mathematician! Use your tools to solve these "
        "math problems.",
        messages=messages,
        tools=tools,
        response_model=ToolCallOrAnswer
    )

    print("Initial Response:")
    print(f"Message: {message}")

    # # serializes response into a python dictionary
    # print(message.model_dump())

    # loop while the agent needs to keep using tools
    while message.needs_tool:
        tool_name = message.tool_name
        tool_input = message.tool_input
        tool_use_id = str(uuid.uuid4())  # generate random tool use id

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        messages.append({
            "role": "assistant",
            "content": {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": tool_input
            }
        })

        tool_result = process_tool_call(tool_name, tool_input)
        print(f"Tool Result: {tool_result}")

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

        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=messages,
            tools=tools,
            response_model=ToolCallOrAnswer
        )

        print(message)

    else:  # else executes after the while loop finishes
        # Get the final response text from the last message
        response = message.final_answer
        print(f"Final Answer: {response}\n")
        return response


chat_with_claude("Calculate ((1 + 2) * 15) / 3 - 18")
chat_with_claude("Calculate ((123 + 456) / 789) + (1000 * 987)")

import re
import anthropic
import os
from anthropic.types import Message
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")

anthropic_client = anthropic.Anthropic()

def claude(user_prompt: str, system_prompt="", model=MODEL_NAME, tools=[]) -> str:
    """
    Calls Claude with the given user prompt, system prompt, and tools (if provided)
    Returns list of MessageBlock or a ToolUseBlock
    """
    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response: Message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        temperature=0.7,
        messages=messages,
        tools=tools if tools else anthropic.NOT_GIVEN
    )

    print(f"Claude says: {response}")
    if response.stop_reason == "tool_use":
        return response.content
    else:  # else return the text of Claude's message
        return response.content[0].text

# define default schema for response
class Response(BaseModel):
    text: str

# util to call Claude with structured output!
# when we wrap Claude with instructor, Claude will automatically perform tool calls and return the response after making tool calls
def structured_claude(user_prompt: str, response_model=Response, system_prompt="", model=MODEL_NAME, tools=[], parallel_tools=False) -> str:
    """
    Util to call Claude with structured output!
    Wraps Claude with the instructor library.
    Claude is capable of making multiple tools calls, but make sure the response_model is an Iterable
    """

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    if parallel_tools:
        structured_client = instructor.from_anthropic(anthropic_client, mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS)
    else:
        structured_client = instructor.from_anthropic(anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS)
    response: Message = structured_client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        temperature=0.7,
        messages=messages,
        response_model=response_model,
        tools=tools if tools else None
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

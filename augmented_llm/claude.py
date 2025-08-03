import anthropic
import instructor
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

anthropic_client = anthropic.Anthropic()

# response without structured output
message = anthropic_client.messages.create(
    model=os.getenv("CLAUDE_MODEL"),
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond in short poems.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why is the ocean salty?"}],
        }
    ],
)

print(message)
print(message.content[0].text)


# wrap anthropic client with instructor to support structured output
client = instructor.from_anthropic(anthropic_client)

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
            "content": [{"type": "text", "text": "Why is the ocean salty?"}],
        }
    ],
    response_model=Response,
)

print(message)
print(message.text)

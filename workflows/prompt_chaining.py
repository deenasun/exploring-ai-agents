import sys
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Add the project root to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import simple_claude, structured_claude
import json

# Building Effective AI Agents: demonstrating the prompt-chaining workflow with a chain that:
# - First determines whether an app/software is open-source or not
# - Performs a gate-check
# - Retrieves documentation for the app/software
# - Writes a summary!

open_source_prompt ="""
You are a helpful researcher.
You are given the name of an app or software.
Your task is to determine whether or not this app or software has an open-source, publicly available GitHub repo.
If it is, return Yes and the url of the GitHub repo.
Otherwise, return No and leave the url blank.

Examples:
Software name: Cybench
Response: {
    "open_source": "Yes",
    "github_url": "https://github.com/andyzorigin/cybench"
}

Software name: Git
Response: {
    "open_source": "Yes",
    "github_url": "https://github.com/git/git"
}

Software name: Google
Response: {
    "open_source": "No",
    "github_url": None
}

Return only the response in exactly this format. Do not include any text before or after.
"""

documentation_prompt="""
You are a helpful software engineer.
You are given the GitHub repo url of an open-source app or software.
Please use the GitHub repo to find some websites about this app/software that would be helpful for creating a summary about this technology.
If you can, please try to find links to official documentation.
Return a list of 3 of the most relevant urls.

Examples:
Software name: Cybench
Response: ["https://cybench.github.io/", "https://arxiv.org/abs/2408.08926", "https://crfm.stanford.edu/2024/08/19/cybench.html"]

Return only the response in exactly this format. Do not include any text before or after.
"""

summarizer_prompt="""
You are a helpful software engineer.
You are given the GitHub repo url of an open-source app or software.
You are also given a list of 3 helpful pieces of online documentation about this technology.
Please use the information from the GitHub repo and the 3 urls to summarize what this app or software does.

If possible, please include information about this technology's architecture in your summary, e.g.
- what language is this technology coded in?
- what servers, databases, apps, APIs, etc. does this technology require?
"""

def prompt_chaining(initial_message: str):
    print(f"Initial message: {initial_message}")

    open_source_response = simple_claude(open_source_prompt, initial_message)

    print(f"Response from the first LLM call in the chain: {open_source_response}")

    gate_input = json.loads(open_source_response)
    if gate_input.get("open_source", "No") == "No":
        return "Exit: this technology is not open source."
    
    github_repo = gate_input.get("github_url", None)
    
    helpful_urls = simple_claude(documentation_prompt, github_repo)
    print(f"Response from the second LLM call in the chain: {helpful_urls}")

    summarizer_response = simple_claude(summarizer_prompt, f"GitHub repo: {github_repo}\nHelpful urls: {helpful_urls}")
    print(f"Response from the third LLM call in the chain: {summarizer_response}")
    return summarizer_response


initial_message="Is Vim open-source?"

summary = prompt_chaining(initial_message)

# Observations: in scenarios where the next steps in the chain depend on the formatting of the outputs from previous steps
# it's very hard to get the outputs exactly right---even with careful prompting!

# would probably work better with structured outputs. so let's try it out!
class OpenSourceResponse(BaseModel):
    open_source: Literal["Yes", "No"] = Field(description="True if the app/software is open-source, False otherwise")
    github_url: Optional[str] = Field(description="Url of this technology's GitHub repo. Blank if the technology is not open-source")


class UrlResponse(BaseModel):
    urls: list[str] = Field(description="A list of 3 helpful urls for learning more about this technology")


class SummaryResponse(BaseModel):
    text: str = Field(description="A summary of this app/software")


def structured_prompt_chaining(initial_message: str):
    print(f"Initial message: {initial_message}")

    open_source_response = structured_claude(initial_message, OpenSourceResponse, system_prompt=open_source_prompt)

    print(f"Response from the first LLM call in the chain: {open_source_response}")

    if open_source_response.open_source == "No":
        return "Exit: this technology is not open source."
    
    github_repo = open_source_response.github_url
    
    documentation_response = structured_claude(github_repo, UrlResponse, system_prompt=documentation_prompt)
    print(f"Response from the second LLM call in the chain: {documentation_response}")

    summarizer_response = structured_claude(f"GitHub repo: {github_repo}\nHelpful urls: {documentation_response.urls}", SummaryResponse, system_prompt=summarizer_prompt)
    print(f"Response from the third LLM call in the chain: {summarizer_response}")
    return summarizer_response


initial_message="Joplin"

summary = structured_prompt_chaining(initial_message)
print(summary.text)
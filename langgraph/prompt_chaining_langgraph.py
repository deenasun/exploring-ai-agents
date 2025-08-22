import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")
model = init_chat_model(MODEL_NAME, model_provider="anthropic")

# Define state
class State(TypedDict):
    query: str
    open_source: bool
    github_repo: str
    helpful_urls: list[str]
    summary: str = Field(
        description="Summarized report about this app/software, using the provided GitHub repo and helpful urls"
    )


# Nodes
def open_source_llm(state: State):
    class OpenSourceResponse(BaseModel):
        open_source: bool = Field(
            description="True if the app/software is open-source and False otherwise"
        )
        github_repo: str = Field(
            default="",
            description="The url of this app/software's GitHub repo if it is open source. Leave the url blank if the app/software is not open-source",
        )

    system_prompt = """
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
    """
    system_message = SystemMessage(content=system_prompt)
    query = state["query"]
    user_message = HumanMessage(content=query)
    messages = [system_message, user_message]

    structured_model = model.with_structured_output(OpenSourceResponse)
    response = structured_model.invoke(messages)

    return {"open_source": response.open_source, "github_repo": response.github_repo}


# Conditional edge function to check if the technology is open source
def is_open_source_gate(state: State):
    """Gate node to check if the technology is open source"""

    if state["open_source"] and state["github_repo"]:
        return "Pass"
    else:
        return "Fail"


def helpful_url_llm(state: State):
    class HelpfulUrlResponse(BaseModel):
        urls: list[str] = Field(
            description="3 helpful urls to documentation or tutorials about this app/software"
        )

    system_prompt = """
You are a helpful software engineer.
You are given the GitHub repo url of an open-source app or software.
Please use the GitHub repo to find some websites about this app/software that would be helpful for creating a summary about this technology.
If you can, please try to find links to official documentation.
Return a list of 3 of the most relevant urls and DO NOT include the original GitHub repo you were provided in your response.

Examples:
Software name: Cybench, github_repo: "https://github.com/andyzorigin/cybench"
Response: ["https://cybench.github.io/", "https://arxiv.org/abs/2408.08926", "https://crfm.stanford.edu/2024/08/19/cybench.html"]
    """
    system_message = SystemMessage(content=system_prompt)
    query = state["query"]
    github_repo = state["github_repo"]

    user_message = HumanMessage(content=f"{query}. GitHub Repo: {github_repo}")
    messages = [system_message, user_message]

    structured_model = model.with_structured_output(HelpfulUrlResponse)
    response = structured_model.invoke(messages)

    return {"helpful_urls": response.urls}


def summarizer_llm(state: State):
    system_prompt = """
You are a helpful software engineer.
You are given the GitHub repo url of an open-source app or software.
You are also given a list of 3 helpful pieces of online documentation about this technology.
Please use the information from the GitHub repo and the 3 urls to summarize what this app or software does.

If possible, please include information about this technology's architecture in your summary, e.g.
- what language is this technology coded in?
- what servers, databases, apps, APIs, etc. does this technology require?
"""
    system_message = SystemMessage(content=system_prompt)
    github_repo = state["github_repo"]
    helpful_urls = state["helpful_urls"]

    message=f"""
Github Repo: {github_repo}
Helpful urls: {helpful_urls}
"""
    user_message = HumanMessage(content=message)
    messages = [system_message, user_message]

    response = model.invoke(messages)
    return {"summary": response.content}

builder = StateGraph(state_schema=State)

# add nodes to graph
builder.add_node("open_source_llm", open_source_llm)
builder.add_node("helpful_url_llm", helpful_url_llm)
builder.add_node("summarizer_llm", summarizer_llm)

# connect nodes with edges
builder.add_edge(START, "open_source_llm")

# add conditional edge
# first param is the start node
# second param is a callable, which can return any hashable object (e.g. string) or the names of nodes to go to next (e.g. END)
# third param is a path_map (optional), which is a dictionary that specifies which node to go to next depending on the output from the callable
builder.add_conditional_edges(
    "open_source_llm",
    is_open_source_gate,
    {
        "Pass": "helpful_url_llm",
        "Fail": END
    }
)

builder.add_edge("helpful_url_llm", "summarizer_llm")
builder.add_edge("summarizer_llm", END)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

query = "Is vim open source?"
state = graph.invoke({"query": query})

print(f"Initial Query: {query}")
print("\n---\n")
print(f"Open source? {state["open_source"]}")
print("\n---\n")
if "github_repo" in state and state["github_repo"]:
    print(f"GitHub repo: {state["github_repo"]}")
    print("\n---\n")
if "helpful_urls" in state:
    print(f"Helpful urls: {state["helpful_urls"]}")
    print("\n---\n")
if "summary" in state:
    print(f"Summary: {state["summary"]}")

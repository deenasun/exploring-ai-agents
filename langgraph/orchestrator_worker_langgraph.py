import sys
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display, Markdown
from langgraph.types import Send
from pydantic import BaseModel, Field
import operator

# Add the project root to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_pdf_text


load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")
model = init_chat_model(MODEL_NAME, model_provider="anthropic", max_tokens=4096)


class Task(BaseModel):
    instructions: str = Field(
        description="Instructions for the sub-agent, e.g. a brief overview of the paper and instructions on how to summarize their section"
    )
    name: str = Field(description="Name of the section in the research paper")
    content: str = Field(
        description="Content of a section copied directly from the paper"
    )


class Tasks(BaseModel):
    tasks: list[Task] = Field(description="List of tasks for each section in the paper")


class State(TypedDict):
    paper_url: str  # url of the paper
    paper: str  # text from the paper
    tasks: Tasks  # list of sections to summarize
    completed_tasks: Annotated[
        list, operator.add
    ]  # all workers write to this key in parallel
    final_summary: str


class WorkerState(TypedDict):
    task: Task
    completed_tasks: Annotated[
        list, operator.add
    ]  # when workers write to this key, they also write to the overall State


def orchestrator(state: State):
    """Orchestrator to break down the paper into different tasks"""
    paper = state.get("paper", "")
    if not paper:
        raise ValueError("'paper' not in the state dictionary")

    system_prompt = """
You are an expert researcher.
You will be orchestrating a team of other researchers to help summarize a paper.
Given the contents of a research paper, break up the paper into its main sections.
Delegate each section to another researcher, who will be in charge of summarizing the section's content.

Your final output should consist of a list of tasks for each researcher.
The list may have a variable length depending on how many sections the paper has.
Each task should include the following:
* Instructions for the other researcher (i.e. summarize the research paper and tell them they are trying to summarize a certain section)
* Section name
* Content from that section from the paper.

The content should be copied directly from the paper itself.
"""
    user_prompt = f"""
Here is the entire content of the paper you will be summarizing:
<research_paper>
{paper}
</research_paper>
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    orchestrator_llm = model.with_structured_output(Tasks)

    response = orchestrator_llm.invoke(messages)

    print(f"Response from the orchestrator llm: {response}\n")

    return {"tasks": response.tasks}


def worker(state: WorkerState) -> dict:
    """Worker node summarizes an individual section of the paper"""

    task = state["task"]

    system_prompt = """
You are a helpful researcher.
You are working with a team of other researchers to summarize a paper.
Given one section of a research paper, please write a clear and detailed summary of your section.

Please also add the section title in markdown bold before your summary.

In your response, do not include any preface before your summary. Only output your final summary.

Example:
**Model Architecture**
... your summary ...
"""

    user_prompt = f"""
Here are your instructions:
{task.instructions}
{task.name}
{task.content}
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    response = model.invoke(messages)

    print(f"Response from the worker llm: {response}\n")

    # write the summary to the completed sections in the overall graph state
    return {"completed_tasks": [response.content]}


def assign_workers(state: State):
    """Assign a worker to each task in the plan"""

    print(f"Assigning workers for the {len(state['tasks'])} tasks in the plan")

    # initiate all the workers in parallel using Langgraph's Send
    return [Send("worker_node", {"task": task}) for task in state["tasks"]]


def aggregator(state: State) -> State:
    """Aggregates all the completed sections"""
    completed_tasks = state["completed_tasks"]
    final_summary = f"\n\n{'=' * 50}\n\n".join(completed_tasks)
    return {"final_summary": final_summary}


def get_pdf_text_node(state: State) -> State:
    paper_url = state["paper_url"]
    paper = get_pdf_text(pdf_url=paper_url)  # fetch content of the paper
    return {"paper": paper}


def orchestrator_worker(user_input: str):
    graph_builder = StateGraph(State)

    graph_builder.add_node("pdf_node", get_pdf_text_node)
    graph_builder.add_node("orchestrator_node", orchestrator)
    graph_builder.add_node("worker_node", worker)
    graph_builder.add_node("aggregator_node", aggregator)

    graph_builder.add_edge(START, "pdf_node")
    graph_builder.add_edge("pdf_node", "orchestrator_node")
    graph_builder.add_conditional_edges(
        "orchestrator_node", assign_workers, ["worker_node"]
    )
    graph_builder.add_edge("worker_node", "aggregator_node")
    graph_builder.add_edge("aggregator_node", END)

    graph = graph_builder.compile()

    final_state = graph.invoke({"paper_url": user_input})

    final_summary = final_state["final_summary"]

    # print mermaid png of graph and final summary in Jupyter notebook environments
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    #     Markdown(final_summary)
    # except Exception:
    #     pass

    print("=" * 50)
    print("Final Summary\n")
    print("=" * 50)
    print(final_summary)
    print("=" * 50)

    return final_state


if __name__ == "__main__":
    print("Hello world!")
    user_input = input(
        "What research paper would you like to summarize? Input the url of the paper's pdf: "
    )

    orchestrator_worker(user_input)

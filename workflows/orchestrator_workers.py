import sys
import os

# Add the project root to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import claude, get_pdf_text
import asyncio
from concurrent.futures import ThreadPoolExecutor

# download a research paper, read it, create writers for each section
attention_is_all_you_need = "https://arxiv.org/pdf/1706.03762"


def orchestrator(paper: str):
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

Separate each task with this special character: <TASK_DIVIDER>

An example of a task:
<Instructions>
"Attention is All You Need" is a machine learning paper that introduced the transformer architecture.
You are a helpful researcher who needs to summarize the Introduction of the paper "Attention is All You Need".
</Instructions>

<Section Name>
Introduction
</Section Name>

<Content>
1 Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks
in particular, have been firmly established as state of the art approaches in sequence modeling and
transduction problems such as language modeling and machine translation [35, 2, 5]...
</Content>
"""
    user_prompt = f"""
Here is the entire content of the paper you will be summarizing:
<research_paper>
{paper}
</research_paper>
"""
    response = claude(
        user_prompt=user_prompt, system_prompt=system_prompt, max_tokens=4096
    )

    # print(f"Response from the orchestrator: {response}")

    tasks = response.content[0].text.split("<TASK_DIVIDER>")
    tasks = [task.strip() for task in tasks if "<Instructions>" in task]

    print(f"Orchestrator created {len(tasks)} tasks.\n")
    return tasks


def worker(instructions: str) -> str:
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
    response = claude(user_prompt=instructions, system_prompt=system_prompt)

    # print(f"Response from the worker: {response}")

    return response.content[0].text


async def orchestrator_worker(paper_url: str):
    # paper = get_pdf_text(paper_url)
    paper = get_pdf_text(attention_is_all_you_need)

    tasks = orchestrator(paper)

    summaries = []

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [loop.run_in_executor(executor, worker, task) for task in tasks]

        results = await asyncio.gather(*tasks)
        summaries.extend(results)

    for i, summary in enumerate(summaries):
        print("=" * 50)
        print(f"Section {i + 1}")
        print(summary)
        print("=" * 50, "\n")


if __name__ == "__main__":
    print("Hello world!")
    user_input = input(
        "What research paper would you like to summarize? Input the url of the paper's pdf: "
    )
    asyncio.run(orchestrator_worker(paper_url=user_input))

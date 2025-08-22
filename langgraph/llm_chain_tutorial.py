import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
MODEL_NAME = os.getenv("CLAUDE_MODEL")
model = init_chat_model(MODEL_NAME, model_provider="anthropic")

messages=[
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

ai_message = model.invoke(messages)

# these are all equivalent ways of calling a model!
# model.invoke("Hello")

# model.invoke([{"role": "user", "content": "Hello"}])

# model.invoke([HumanMessage("Hello")])

# Streaming -> the stream method returns AIMessageChunk objects
for token in model.stream(messages):
    print(token.content, end="")

from langgraph.graph import START, END, StateGraph, add_messages

# Simple chatbot graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(state_schema=State)

def call_model(state: State):
    response = model.invoke(state["messages"])
    return {"messages": response}

builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

# compile the graph
graph = builder.compile()

# show the graph!
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

message = [
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

# call the graph by passing in the initial state
final_state = graph.invoke({"messages": message})
print(final_state)

# Chatbot with memory state
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Messages state is a state that comes predefined with a messages (type: list of messages) and message reducers

memory_graph_builder = StateGraph(state_schema=MessagesState)

def call_model(state: State):
    response = model.invoke(state["messages"])
    return {"messages": response}

memory_graph_builder.add_node("model", call_model)
memory_graph_builder.add_edge(START, "model")
memory_graph_builder.add_edge("model", END)

# Add memory
memory = MemorySaver()

# compile the graph
memory_graph = memory_graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

message = [
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

output = memory_graph.invoke({"messages": message}, config)

for message in output["messages"]:
    message.pretty_print()

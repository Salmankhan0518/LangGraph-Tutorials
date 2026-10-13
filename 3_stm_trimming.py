from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from langchain_core.messages.utils import trim_messages, count_tokens_approximately


load_dotenv()

model = ChatOllama(model="qwen2.5:3b")

MAX_TOKENS = 150

def call_model(state: MessagesState):

    # Trim conversation history -> last N messages that fit within the token budge
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    print("Current Token count ->", count_tokens_approximately(messages=messages))

    for message in messages:
        print(message.content)

    response = model.invoke(messages)

    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "chat-1"}}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Salman"}]},
    config,
)

result["messages"][-1].content


result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "I am learning LangGraph."}]},
    config,
)

print("--result1--")
result1["messages"][-1].content


result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "Can you explain short term memory?"}]},
    config,
)

print("--result2--")
result2["messages"][-1].content


result3 = graph.invoke(
    {"messages": [{"role": "user", "content": "Can you explain AI taking jobs or not?"}]},
    config,
)

print("--result3--")
result3["messages"][-1].content
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="qwen2.5:3b")

def call_model(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    #Run ONCE (create tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    #Thread 1 (remember)
    t1 = {"configurable": {"thread_id": "thread-1"}}

    output1 = graph.invoke({"messages": [{"role": "user", "content": "Hi My name is Salman"}]}, t1)
    output2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, t1)

    print("--- Output 1 ---")
    print(output1["messages"][-1].content)
    print("--- Output 2 ---")
    print(output2["messages"][-1].content)


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 2 (fresh)
    t2 = {"configurable": {"thread_id": "thread-2"}}
    out2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t2)
    print("--- Output of Thread 2 ---")
    print("Thread-2:", out2["messages"][-1].content)


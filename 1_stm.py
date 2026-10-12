from langgraph.graph import StateGraph, START, MessagesState
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


load_dotenv()

model = ChatOllama(model='qwen2.5:3b')

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile()

output1 = graph.invoke({"messages": [{"role": "user", "content": "Hi My name is Salman"}]})
output2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]})

print("--- Output 1 ---")
print(output1["messages"][-1].content)

print("--- Output 2 ---")
print(output2["messages"][-1].content)
import sys
import uuid

# uuid_utils monkeypatching
sys.modules['uuid_utils'] = uuid
sys.modules['uuid_utils.compat'] = uuid

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# 1. State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. LLM Model Initialization
# (Model name specific rakhein jaise 'qwen2.5:3b' ya 'llama3.2')
llm = ChatOllama(model="qwen2.5:3b")

# 3. Node Function Fix
def chat_node(state: ChatState):
    messages = state['messages']
    # FIX: Class ki bajaye 'llm' instance ko invoke karein
    response = llm.invoke(messages)
    return {'messages': [response]}

# 4. Graph Construction
checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# 5. Thread Configuration
thread_id = '1'
config = {'configurable': {'thread_id': thread_id}}

# 6. Interactive Chat Loop
while True:
    user_message = input("Type here: ")

    if user_message.strip().lower() in ['exit', 'quit', 'bye', 'end']:
        print("Goodbye!")
        break


    # FIX: Key ka naam 'messages' pass karein ('message' nahi)
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_message)]}, 
        config=config
    )

    print('AI:', response['messages'][-1].content)
    print("-" * 40)
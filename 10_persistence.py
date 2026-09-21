import sys
import uuid
from typing import TypedDict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# Monkeypatching for uuid_utils compatibility
sys.modules['uuid_utils'] = uuid
sys.modules['uuid_utils.compat'] = uuid

# Model setup
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    prompt = f'generate a short joke on the topic: {state["topic"]}'
    response = llm.invoke(prompt).content
    return {'joke': response}

def generate_explanation(state: JokeState):
    prompt = f'write a short explanation for the joke - {state["joke"]}'
    response = llm.invoke(prompt).content
    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# Step 1: Initial Graph Execution
config1 = {"configurable": {"thread_id": "1"}}
print("=== Initial Workflow Execution ===")
result1 = workflow.invoke({'topic': 'pizza'}, config=config1)
print(result1)

# Step 2: Get History / Checkpoints
print("\n=== Checkpoint History ===")
history = list(workflow.get_state_history(config1))
for state in history:
    print(f"Checkpoint ID: {state.config['configurable']['checkpoint_id']} | Next Node: {state.next}")

# Root/Start State Object
start_state = history[-1]

print(f"\n=== Getting State at Checkpoint: {start_state.config['configurable']['checkpoint_id']} ===")
print("Values at start:", start_state.values)

# FIX: `as_node` hata diya hai taaki `generate_joke` node dubara run ho
print("\n=== Updating State ===")
updated_config = workflow.update_state(
    config=start_state.config, 
    values={'topic': 'samosa'}
)

# Step 3: Resume Execution from updated checkpoint configuration
print("\n=== Resuming Workflow from Updated Checkpoint ===")
resumed_result = workflow.invoke(None, config=updated_config)
print(resumed_result)
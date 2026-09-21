from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

class BlogState(TypedDict):

    title: str
    outline: str
    content: str


def create_outline(state: BlogState):
    title = state['title']
    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content
    
    # Poora state return karne ke bajaye sirf updated field return karein
    return {'outline': outline}

def create_blog(state: BlogState):
    title = state['title']
    outline = state['outline']
    prompt = f'Write a detailed blog on the title - {title} using the following outline \n {outline}'
    content = model.invoke(prompt).content
    
    return {'content': content}


graph = StateGraph(BlogState)

# nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()
initial_state = {'title': 'Rise of AI in Pakistan', 'outline': '', 'content': ''}

final_state = workflow.invoke(initial_state)

print(final_state)

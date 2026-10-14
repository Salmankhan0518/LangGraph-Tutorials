from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START
from langgraph.store.memory import InMemoryStore


load_dotenv()

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

store = InMemoryStore(index={'embed': embedding_model, 'dims': 1536})

# creating a namespace
namespace = ("user", "u1")

store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})


items = store.search(namespace, query="what is user currently learning", limit=1)

for item in items:
    print(item.value)


items = store.search(namespace, query="what are user's preferences", limit=3)

for item in items:
    print(item.value)    
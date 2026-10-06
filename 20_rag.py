from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOllama(model="qwen2.5:3b")

loader = PyPDFLoader("intro-to-ml.pdf")
docs = loader.load()

print(len(docs))






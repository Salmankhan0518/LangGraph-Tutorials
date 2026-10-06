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
import os

load_dotenv()

llm = ChatOllama(model="qwen2.5:3b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

DB_PATH = "faiss_index"

if os.path.exists(DB_PATH):
    print("Loading existing FAISS store...")
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS store...")
    loader = PyPDFLoader("intro-to-ml.pdf")
    docs = loader.load()
    print(len(docs))


splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(len(chunks))


vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type= 'similarity', search_kwargs={'k':4})


@tool
def rag_tool(query):

    """
    Retrieve relevent information from the pdf document.
    Use this tool when the user asks factual / conceptual questions 
    that might be answered from the stored documents.
    """

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]


    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }

tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}


tool_node = ToolNode(tools)


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile()

print(chatbot)


result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Using the pdf notes, explain how to find the ideal value of K in KNN"
                )
            )
        ]
    }
)

print(result['messages'][-1].content)
import sqlite3
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Safely check Streamlit secrets without crashing locally
groq_key = None
try:
    groq_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

GROQ_API_KEY = groq_key or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please add it to .env locally or Streamlit Cloud Secrets.")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

graph = StateGraph(ChatState)

# Persistent SQLite database checkpointer
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
check_pointer = SqliteSaver(connection)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compiled workflow export
workflow = graph.compile(checkpointer=check_pointer)

def retrive_all_thread():
    """Retrieve unique thread IDs sorted by activity."""
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        thread_id = checkpoint.config['configurable'].get('thread_id')
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)

# Save graph visualization to PNG
try:
    png_data = workflow.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph image saved successfully as graph.png")
except Exception as e:
    print(f"Failed to generate graph PNG: {e}")
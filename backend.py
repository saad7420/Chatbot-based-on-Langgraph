import os
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from tools import tools, llm_with_tools

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant with access to tools. "
        "When calling a tool, generate ONLY valid JSON matching the tool function arguments. "
        "Do not write conversational text or explanation when generating a tool call. "
        "If no tool is required, answer the user directly."
    )
)

def chat_node(state: ChatState):
    messages = state['messages']
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
check_pointer = SqliteSaver(connection)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

workflow = graph.compile(checkpointer=check_pointer)

def retrive_all_thread():
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        thread_id = checkpoint.config['configurable'].get('thread_id')
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)
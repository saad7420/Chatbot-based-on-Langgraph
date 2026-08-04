import uuid
import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage
from backend import retrive_all_thread, workflow

# Page Config
st.set_page_config(page_title="Chat Workspace", layout="wide")

st.markdown(
    """
    <style>
    /* 1. Overall Layout & Spacing */
    .block-container { 
        padding-top: 1.8rem; 
        padding-bottom: 2rem; 
        max-width: 900px;
    }

    /* 2. Sidebar Text & Header Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 1rem !important;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        opacity: 0.7;
    }

    /* 3. Sidebar Conversation Buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        text-align: left;
        border: none;
        padding: 0.45rem 0.75rem;
        background: transparent;
        font-size: 1rem !important;
        font-weight: 400;
        line-height: 1.4;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: background-color 0.15s ease-in-out;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(128, 128, 128, 0.15);
    }

    /* 4. Main Chat Typography & Bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Helper Functions
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.session_state["messages_history"] = []

def load_conv(thread_id):
    state = workflow.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    if state and state.values:
        return state.values.get("messages", [])
    return []

def get_thread_preview(thread_id):
    """Retrieve thread title from initial user message or state."""
    if thread_id in st.session_state.get("thread_titles", {}):
        return st.session_state["thread_titles"][thread_id]
    
    messages = load_conv(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            title = content[:28] + "..." if len(content) > 28 else content
            st.session_state["thread_titles"][thread_id] = title
            return title
            
    return "New Conversation"

# Initialize Session States
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrive_all_thread()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# Ensure active thread exists in list
if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(st.session_state["thread_id"])

# Sidebar Layout
with st.sidebar:
    st.title("Assistant")
    st.button("New chat", on_click=reset_chat, use_container_width=True)
    st.markdown("---")
    st.caption("Recent chats")

    for thread_id in st.session_state["chat_threads"][::-1]:
        title = get_thread_preview(thread_id)
        is_active = thread_id == st.session_state["thread_id"]
        display_title = f"• {title}" if is_active else title
        
        if st.button(display_title, key=f"btn_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_conv(thread_id)
            
            st.session_state["messages_history"] = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                st.session_state["messages_history"].append({
                    "role": role,
                    "content": msg.content
                })
            st.rerun()

# Main Workspace
st.subheader("How can I help you today?")

# Render Historical Messages
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Streaming Handler
user_input = st.chat_input("Type a message...")

if user_input:
    current_thread = st.session_state["thread_id"]
    
    if current_thread not in st.session_state["thread_titles"]:
        summary_title = user_input[:28] + "..." if len(user_input) > 28 else user_input
        st.session_state["thread_titles"][current_thread] = summary_title

    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": current_thread}}

    with st.chat_message("assistant"):
        def stream_with_status():
            status_container = None
            
            for event in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(event, tuple):
                    message_chunk, metadata = event
                    
                    if isinstance(message_chunk, AIMessageChunk) and message_chunk.tool_calls:
                        for tool_call in message_chunk.tool_calls:
                            tool_name = tool_call.get("name", "tool")
                            status_container = st.status(
                                f"Using **{tool_name}** tool...", 
                                state="running", 
                                expanded=True
                            )
                            status_container.write(f"Executing `{tool_name}` with arguments: `{tool_call.get('args')}`")

                    elif message_chunk.__class__.__name__ == "ToolMessage":
                        if status_container:
                            status_container.write("Tool execution completed.")
                            status_container.update(
                                label="Tool execution completed!", 
                                state="complete", 
                                expanded=False
                            )

                    elif isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                        yield message_chunk.content

        assistant_response = st.write_stream(stream_with_status())

    st.session_state["messages_history"].append({
        "role": "assistant",
        "content": assistant_response
    })
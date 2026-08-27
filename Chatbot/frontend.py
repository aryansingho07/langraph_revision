import streamlit as st
from langgraph_tool_rag import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ============================================ Page Config ==========================================
st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================ Custom CSS ============================================
st.markdown(
    """
    <style>
    /* ---- App background ---- */
    .stApp { background-color: #0e1117; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
        padding: 0.5rem 1rem;
        text-align: left;
        transition: background-color 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
    }

    /* ---- Chat messages ---- */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* ---- Tool call badge ---- */
    .tool-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151; border-radius: 20px;
        padding: 6px 14px; margin: 4px 2px; font-size: 0.82rem;
        color: #9ca3af; font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .tool-badge .icon { font-size: 1rem; }
    .tool-badge .name { color: #d1d5db; font-weight: 500; }

    /* ---- Document info card ---- */
    .doc-card {
        background: linear-gradient(135deg, #1a2332, #162032);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 16px 20px; margin: 8px 0; color: #c9d1d9;
    }
    .doc-card h4 { margin: 0 0 8px 0; color: #58a6ff; font-size: 0.95rem; }
    .doc-card .stat {
        display: inline-block; margin-right: 16px;
        font-size: 0.85rem; color: #8b949e;
    }
    .doc-card .stat strong { color: #c9d1d9; }

    /* ---- Header area ---- */
    .header-area {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d; border-radius: 16px;
        padding: 24px 32px; margin-bottom: 20px; text-align: center;
    }
    .header-area h1 { margin: 0; color: #f0f6fc; font-size: 1.8rem; }
    .header-area p { margin: 6px 0 0; color: #8b949e; font-size: 0.95rem; }

    /* ---- Feature pills ---- */
    .feature-pill {
        display: inline-block; background-color: #21262d;
        border: 1px solid #30363d; border-radius: 16px;
        padding: 4px 12px; margin: 3px; font-size: 0.78rem; color: #8b949e;
    }

    /* ---- Empty state ---- */
    .empty-state { text-align: center; padding: 60px 20px; color: #484f58; }
    .empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
    .empty-state h3 { color: #8b949e; margin-bottom: 8px; }
    .empty-state p { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================ Utility Functions =====================================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["doc_info"] = None


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def format_tool_badge(tool_name):
    icons = {
        "search": "🔍",
        "duckduckgo_search": "🔍",
        "calculator": "🧮",
        "get_stock_price": "📈",
        "rag_tool": "📚",
    }
    icon = icons.get(tool_name, "⚙️")
    pretty = tool_name.replace("_", " ").title()
    return f'<span class="tool-badge"><span class="icon">{icon}</span><span class="name">{pretty}</span></span>'


# ============================================ Session State =========================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "doc_info" not in st.session_state:
    st.session_state["doc_info"] = None

add_thread(st.session_state["thread_id"])

# ============================================ Sidebar ==============================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 8px 0 16px;">'
        '<span style="font-size:1.4rem;">🤖</span> '
        '<span style="font-size:1.1rem; font-weight:600; color:#f0f6fc;"> LangGraph Chat</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("✨ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("##### 📎 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF for RAG",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        with st.spinner("Indexing document..."):
            result = ingest_pdf(
                uploaded_file.getvalue(),
                st.session_state["thread_id"],
                uploaded_file.name,
            )
            st.session_state["doc_info"] = result
            st.rerun()

    if thread_has_document(st.session_state["thread_id"]):
        meta = thread_document_metadata(st.session_state["thread_id"])
        st.markdown(
            f'<div class="doc-card">'
            f'<h4>📄 Document Loaded</h4>'
            f'<span class="stat"><strong>{meta.get("filename", "unknown")}</strong></span><br/>'
            f'<span class="stat">Pages: <strong>{meta.get("documents", 0)}</strong></span> '
            f'<span class="stat">Chunks: <strong>{meta.get("chunks", 0)}</strong></span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 💬 Conversations")
    for tid in reversed(st.session_state["chat_threads"]):
        label = tid[:8] + "..."
        is_active = tid == st.session_state["thread_id"]
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            label, key=f"thread_{tid}", use_container_width=True, type=btn_type
        ):
            st.session_state["thread_id"] = tid
            messages = load_conversation(tid)
            temp = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    temp.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    temp.append({"role": "assistant", "content": msg.content})
            st.session_state["message_history"] = temp
            st.session_state["doc_info"] = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="padding: 8px 0; color: #484f58; font-size: 0.78rem;">'
        '<span class="feature-pill">🧠 Gemini 2.5 Flash</span>'
        '<span class="feature-pill">🔍 Web Search</span>'
        '<span class="feature-pill">🧮 Calculator</span>'
        '<span class="feature-pill">📈 Stock Price</span>'
        '<span class="feature-pill">📚 RAG / PDF</span>'
        "<br/>"
        '<span class="feature-pill">💾 SQLite Memory</span>'
        '<span class="feature-pill">🔀 Tool Routing</span>'
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================ Main Chat Area ========================================
if not st.session_state["message_history"]:
    st.markdown(
        '<div class="header-area">'
        "<h1>🤖 LangGraph AI Assistant</h1>"
        "<p>Your intelligent multi-tool chatbot powered by LangGraph</p>"
        '<div style="margin-top: 12px;">'
        '<span class="feature-pill">🔍 Web Search</span>'
        '<span class="feature-pill">🧮 Calculator</span>'
        '<span class="feature-pill">📈 Stock Prices</span>'
        '<span class="feature-pill">📄 PDF / RAG</span>'
        '<span class="feature-pill">💾 Persistent Memory</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">💬</div>'
        "<h3>Start a conversation</h3>"
        "<p>Ask me anything, upload a PDF for context, or try a tool call!</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Show message history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================ Chat Input ============================================
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.session_state["message_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the graph
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Collect tool calls for display
            tool_calls_display = []
            final_response = ""

            try:
                events = []
                for event in chatbot.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="updates",
                ):
                    events.append(event)

                # Process events to extract tool calls and final response
                for event in events:
                    for node_name, node_data in event.items():
                        if node_name == "tools" and "messages" in node_data:
                            for msg in node_data["messages"]:
                                if isinstance(msg, ToolMessage):
                                    tool_name = msg.name if hasattr(msg, "name") else "unknown"
                                    tool_calls_display.append(tool_name)
                        elif node_name == "chat_node" and "messages" in node_data:
                            for msg in node_data["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    final_response = msg.content

                # Show tool usage badges
                if tool_calls_display:
                    badges_html = "".join(format_tool_badge(t) for t in tool_calls_display)
                    st.markdown(
                        f'<div style="margin-bottom: 8px;">'
                        f'<span style="font-size: 0.78rem; color: #484f58; margin-right: 4px;">Tools used:</span>'
                        f"{badges_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Show final response
                if final_response:
                    st.markdown(final_response)
                else:
                    st.markdown("*No response generated. Try rephrasing your question.*")

            except Exception as e:
                st.error(f"Error: {e}")
                final_response = f"⚠️ Error: {e}"

    # Save assistant response to history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": final_response}
    )
    st.rerun()

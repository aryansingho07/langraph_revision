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

# ============================================ Custom CSS (Premium theme) ============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');

    :root {
        --bg-primary: #0b0d14;
        --bg-secondary: #12141f;
        --bg-elevated: #171a27;
        --border-subtle: #262a3d;
        --accent-primary: #7c6cf6;
        --accent-secondary: #a78bfa;
        --accent-gold: #e8b95f;
        --text-primary: #eef0f7;
        --text-secondary: #9096ab;
        --text-muted: #5c6178;
        --success: #4ade80;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- App background ---- */
    .stApp {
        background: radial-gradient(circle at 20% 0%, #161a2c 0%, #0b0d14 55%);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10121c 0%, #0c0e17 100%);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
        background-color: var(--bg-elevated);
        color: var(--text-secondary);
        padding: 0.55rem 1rem;
        text-align: left;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, rgba(124,108,246,0.18), rgba(167,139,250,0.10));
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-primary), #5b4bd6);
        border: none;
        color: #ffffff;
    }

    /* ---- Chat messages ---- */
    .stChatMessage {
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-subtle);
    }

    /* ---- Tool call badge ---- */
    .tool-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, rgba(232,185,95,0.14), rgba(124,108,246,0.10));
        border: 1px solid rgba(232,185,95,0.35); border-radius: 20px;
        padding: 6px 14px; margin: 4px 4px 4px 0; font-size: 0.8rem;
        color: var(--accent-gold); font-weight: 500;
    }
    .tool-badge .icon { font-size: 1rem; }
    .tool-badge .name { color: var(--text-primary); font-weight: 600; }

    /* ---- Document info card ---- */
    .doc-card {
        background: linear-gradient(135deg, rgba(124,108,246,0.12), rgba(23,26,39,0.9));
        border: 1px solid rgba(124,108,246,0.35); border-radius: 14px;
        padding: 16px 20px; margin: 10px 0; color: var(--text-primary);
        backdrop-filter: blur(6px);
    }
    .doc-card h4 { margin: 0 0 8px 0; color: var(--accent-secondary); font-size: 0.95rem; font-weight: 700; }
    .doc-card .stat {
        display: inline-block; margin-right: 16px;
        font-size: 0.85rem; color: var(--text-secondary);
    }
    .doc-card .stat strong { color: var(--text-primary); }

    /* ---- Header area ---- */
    .header-area {
        background: linear-gradient(135deg, #171a2c 0%, #0d0f1a 100%);
        border: 1px solid var(--border-subtle); border-radius: 20px;
        padding: 32px 36px; margin-bottom: 24px; text-align: center;
        box-shadow: 0 8px 32px rgba(124,108,246,0.08);
    }
    .header-area h1 {
        margin: 0; font-family: 'Sora', sans-serif;
        background: linear-gradient(135deg, #ffffff, var(--accent-secondary));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2rem; font-weight: 700;
    }
    .header-area p { margin: 8px 0 0; color: var(--text-secondary); font-size: 0.98rem; }

    /* ---- Feature pills ---- */
    .feature-pill {
        display: inline-block; background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle); border-radius: 18px;
        padding: 5px 14px; margin: 4px; font-size: 0.8rem; color: var(--text-secondary);
        font-weight: 500;
    }

    /* ---- Empty state ---- */
    .empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
    .empty-state .icon { font-size: 3rem; margin-bottom: 12px; opacity: 0.6; }
    .empty-state h3 { color: var(--text-secondary); margin-bottom: 8px; font-weight: 600; }
    .empty-state p { font-size: 0.9rem; }

    /* ---- Chat input ---- */
    [data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        background-color: var(--bg-elevated);
    }

    /* ---- Divider ---- */
    hr, .stMarkdown hr { border-color: var(--border-subtle) !important; }
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


def extract_text(content) -> str:
    """
    Normalize AIMessage/HumanMessage .content into plain, displayable text.

    Some models (e.g. Gemini via langchain-google-genai) return content as a
    list of content blocks instead of a plain string, e.g.:
        [{'type': 'text', 'text': 'actual answer', 'extras': {'signature': '...'}}]
    This pulls out just the human-readable text and drops signatures/metadata.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Prefer explicit 'text' field; skip non-text blocks (e.g. tool_use, thinking, signatures)
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "\n".join(p.strip() for p in parts if p and p.strip())

    # Fallback: stringify anything unexpected rather than showing a raw repr
    return str(content).strip()


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
        '<span style="font-size:1.15rem; font-weight:700; color:#eef0f7; font-family:\'Sora\',sans-serif;"> LangGraph Chat</span>'
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
                    temp.append({"role": "user", "content": extract_text(msg.content)})
                elif isinstance(msg, AIMessage):
                    text = extract_text(msg.content)
                    if text:
                        temp.append({"role": "assistant", "content": text})
            st.session_state["message_history"] = temp
            st.session_state["doc_info"] = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="padding: 8px 0;">'
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


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ============================================ Main Chat Area ========================================
if not st.session_state["message_history"]:
    st.markdown(
        '<div class="header-area">'
        "<h1>LangGraph AI Assistant</h1>"
        "<p>Your intelligent multi-tool chatbot powered by LangGraph</p>"
        '<div style="margin-top: 14px;">'
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
                                    text = extract_text(msg.content)
                                    if text:
                                        final_response = text

                # Show tool usage badges
                if tool_calls_display:
                    badges_html = "".join(format_tool_badge(t) for t in tool_calls_display)
                    st.markdown(
                        f'<div style="margin-bottom: 10px;">'
                        f'<span style="font-size: 0.78rem; color: #5c6178; margin-right: 6px;">Tools used:</span>'
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
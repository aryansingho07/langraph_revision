# 🤖 LangGraph AI Assistant — Chatbot

A multi-tool AI chatbot built with **LangGraph**, **LangChain**, and **Google Gemini**. This project demonstrates how to build stateful, tool-augmented conversational agents with persistent memory and document retrieval.

---

## 📚 Concepts Covered

This project is a hands-on learning resource for LangGraph. Here's every core concept demonstrated:

### 1. **StateGraph & State Management**
- Defined a `ChatState` TypedDict with `Annotated[list[BaseMessage], add_messages]`
- LangGraph automatically manages how messages accumulate across nodes
- Each conversation has a unique `thread_id` for isolation

### 2. **Nodes (Processing Units)**
- **`chat_node`** — The LLM brain. Receives the full message history, adds a system prompt, invokes the Gemini model with tool bindings
- **`tools`** — A `ToolNode` that executes whichever tool the LLM requested

### 3. **Edges & Routing**
- `START → chat_node` — Every conversation begins at the LLM
- `chat_node → tools` (conditional) — If the LLM wants a tool, route to the ToolNode
- `tools → chat_node` — After tool execution, return to the LLM for a final response
- Uses LangGraph's built-in `tools_condition` for automatic routing

### 4. **Tool Calling (Function Calling)**
Four tools are registered and bound to the LLM:
| Tool | Purpose |
|------|---------|
| 🔍 `DuckDuckGoSearchRun` | Web search for real-time information |
| 🧮 `calculator` | Basic arithmetic (add, sub, mul, div) |
| 📈 `get_stock_price` | Fetches real stock prices via Alpha Vantage API |
| 📚 `rag_tool` | Retrieves context from uploaded PDF documents |

### 5. **RAG (Retrieval-Augmented Generation)**
- Users upload PDFs through the Streamlit UI
- Documents are split into chunks with `RecursiveCharacterTextSplitter`
- Chunks are embedded with `GoogleGenerativeAIEmbeddings` and stored in a **FAISS** vector store
- At query time, the `rag_tool` retrieves the top-4 most relevant chunks
- The LLM uses this context to answer questions about the document

### 6. **Checkpointing & Persistent Memory**
- Uses `SqliteSaver` from `langgraph-checkpoint-sqlite` to persist conversation state
- Every conversation thread is saved to `chatbot.db`
- Users can return to previous conversations — the full message history is restored

### 7. **Multi-threaded Conversations**
- Each chat gets a unique UUID thread ID
- `retrieve_all_threads()` loads all past conversations into the sidebar
- Users can switch between conversations seamlessly

### 8. **Streaming Responses**
- The frontend uses `chatbot.stream()` with `stream_mode="updates"` to show tool usage in real-time
- Tool call badges appear as the agent works, before the final response

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────┐
│  chat_node   │  ← LLM (Gemini 2.5 Flash) + Tool Bindings
└──────┬──────┘
       │
       ├─ No tool needed → Final response to user
       │
       ▼
┌─────────────┐
│    tools     │  ← ToolNode executes the requested tool
└──────┬──────┘
       │
       ▼
   chat_node   │  ← LLM processes tool output, generates response
       │
       ▼
  User Response
```

---

## 📁 Project Structure

```
Chatbot/
├── langgraph_tool_rag.py        # Main backend: LLM, tools, RAG, graph
├── langgraph_database_backend.py # Simpler backend: basic chat (no tools)
├── frontend.py                  # Streamlit UI with professional dark theme
├── chatbot.db                   # SQLite database (auto-created)
├── .env                         # API keys (in parent LangGraph/ directory)
└── README.md                    # This file
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- A Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/))

### 1. Install Dependencies
```bash
pip install langgraph langchain-google-genai langchain-core langchain-community \
    langgraph-checkpoint-sqlite faiss-cpu pypdf streamlit requests python-dotenv
```

### 2. Set Up API Key
Create a `.env` file in the **parent directory** (`LangGraph/`):
```
GOOGLE_API_KEY=your-google-api-key-here
```

### 3. Run the App
```bash
cd Chatbot
python -m streamlit run frontend.py
```

The app opens at **http://localhost:8501**.

---

## 🎯 Usage Tips

- **Ask general questions** — The LLM answers directly
- **Ask for current info** — The search tool fetches real-time web results
- **Upload a PDF** — Use the sidebar uploader, then ask questions about the document
- **Math problems** — "What is 234 × 56?" triggers the calculator
- **Stock prices** — "What's the current price of AAPL?" triggers the stock tool
- **Switch conversations** — Click any thread in the sidebar to resume it

---

## 🧠 What You'll Learn

By studying this code, you'll understand:

1. **How LangGraph graphs work** — Nodes, edges, conditional routing
2. **How tool calling works** — Binding tools to LLMs, the ToolNode pattern
3. **How RAG works** — Document loading, chunking, embedding, retrieval
4. **How persistent state works** — Checkpointing with SQLite
5. **How to build a real UI** — Streamlit with custom CSS and real-time streaming
6. **How to compose these patterns** — Putting it all together into a working app

---

## 📦 Backend Files

### `langgraph_tool_rag.py` (Main)
Full-featured backend with:
- Google Gemini 2.5 Flash LLM
- 4 tools (search, calculator, stock, RAG)
- FAISS vector store for PDF retrieval
- SQLite checkpointing

### `langgraph_database_backend.py` (Simple)
Minimal backend for learning:
- Google Gemini 2.5 Flash LLM
- No tools — just a basic chatbot
- SQLite checkpointing
- Good starting point to understand the basics before adding tools

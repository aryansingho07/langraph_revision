from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated, TypedDict
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("Chat_node", chat_node)

graph.add_edge(START, "Chat_node")
graph.add_edge("Chat_node", END)


conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )
    return list(all_threads)
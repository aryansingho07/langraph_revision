from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    statement: str
    response: str


class Sentiment(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Sentiment of the statement"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

structured_llm = llm.with_structured_output(Sentiment)


def find_sentiment(state: State) -> State:

    result = structured_llm.invoke(state["statement"])

    return {
        "statement": state["statement"],
        "response": result.sentiment
    }


def check_condition(state: State):

    if state["response"] == "Positive":
        return "positive_path"

    elif state["response"] == "Negative":
        return "negative_path"

    else:
        raise ValueError(f"Unknown sentiment: {state['response']}")


def negative_path(state: State) -> State:

    statement = state["statement"]

    res = llm.invoke(
        f"""The following statement has a negative sentiment:

{statement}

Provide practical and constructive advice to improve the situation.
"""
    )

    return {
        "statement": statement,
        "response": res.content
    }


def positive_path(state: State) -> State:

    statement = state["statement"]

    res = llm.invoke(
        f"""The following statement has a positive sentiment:

{statement}

Provide positive and encouraging feedback.
"""
    )

    return {
        "statement": statement,
        "response": res.content
    }


# Create graph
graph = StateGraph(State)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("negative_path", negative_path)
graph.add_node("positive_path", positive_path)


# Edges
graph.add_edge(START, "find_sentiment")

graph.add_conditional_edges(
    "find_sentiment",
    check_condition,
    {
        "positive_path": "positive_path",
        "negative_path": "negative_path"
    }
)

graph.add_edge("negative_path", END)
graph.add_edge("positive_path", END)


workflow = graph.compile()


result = workflow.invoke({
    "statement": "I am  very happy with my performance in my last project."
})


print("Statement:", result["statement"])
print("Response:", result["response"])
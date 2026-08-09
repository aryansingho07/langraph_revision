from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    weight: float
    height: float
    bmi: float
    bmi_llm: float


def cal_bmi(state: State) -> State:
    weight = state["weight"]
    height = state["height"]

    bmi = weight / (height ** 2)

    state["bmi"] = bmi
    return state


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


def bmi_llm(state: State) -> State:
    prompt = f"""
Calculate BMI using:
Weight = {state['weight']} kg
Height = {state['height']} meters

Return ONLY the BMI number.
Do not give any explanation.
"""

    response = llm.invoke(prompt)

    state["bmi_llm"] = float(response.content.strip())

    return state


# Define graph
graph = StateGraph(State)

graph.add_node("bmi_llm", bmi_llm)
graph.add_node("cal_bmi", cal_bmi)

# Add edges
graph.add_edge(START, "bmi_llm")
graph.add_edge("bmi_llm", "cal_bmi")
graph.add_edge("cal_bmi", END)

workflow = graph.compile()


# Input
in_state = {
    "weight": 70.0,
    "height": 1.75
}

res = workflow.invoke(in_state)

print("BMI using normal calculation:", res["bmi"])
print("BMI using LLM:", res["bmi_llm"])
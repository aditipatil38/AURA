from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

load_dotenv()
model = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are AURA, a compassionate and emotionally intelligent virtual therapy assistant. "
                "Always respond briefly (1-3 sentences), using a warm and validating tone. "
                "Your goal is to make the user feel seen, supported, and gently guided. "
                "Use calming language, avoid giving direct advice, and respond in {language}. "
                "If a user is in distress, acknowledge their feelings first. Avoid repeating what the user said verbatim."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

def call_model(state: State):
    return {"messages": [model.invoke(prompt.invoke(state))]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

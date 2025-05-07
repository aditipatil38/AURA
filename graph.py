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
        "You are AURA, a supportive therapy assistant. Respond calmly, warmly, and in {language}."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

def call_model(state: State):
    return {"messages": [model.invoke(prompt.invoke(state))]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

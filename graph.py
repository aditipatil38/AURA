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
from rag import retriever


load_dotenv()
model = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    user_query = state["messages"][-1].content
    docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in docs[:2]])

    prompt_with_context = ChatPromptTemplate.from_messages([
        ("system", f"""You are AURA, a compassionate and emotionally intelligent virtual therapy assistant.
    Use the following context to inform your response:

    {context}

    Always respond briefly (1–3 sentences) in {{language}}, using a warm, validating tone.
    Your goal is to make the user feel seen, supported, and gently guided.
    Use calming language, avoid giving direct advice, and never repeat what the user said verbatim.
    If a user is in distress, first acknowledge their feelings with empathy.
    """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    return {"messages": [model.invoke(prompt_with_context.invoke(state))]}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")


memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

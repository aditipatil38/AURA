from flask import Flask, render_template, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import os

#  Firebase Setup
from dotenv import load_dotenv
import os

load_dotenv()

cred_path = os.getenv("FIREBASE_CREDENTIALS")
db_url = os.getenv("FIREBASE_DB_URL")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {'databaseURL': db_url})
#  Environment Setup
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Retrieve GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Please check your .env file.")

#  LangGraph Setup
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# Initialize model with the API key explicitly
model = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Define the state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Create the stateful graph
workflow = StateGraph(state_schema=State)

# Define the prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
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
    ]
)

# Define the model call function
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Add model node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory checkpointing
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)


#  Flask Setup
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/send", methods=["POST"])
def send_message():
    data = request.json
    user_msg = data["message"]
    language = data.get("language", "English")
    user_id = data.get("user_id", "anon")

    input_messages = [HumanMessage(user_msg)]

    output = app_graph.invoke(
        {"messages": input_messages, "language": language},
        config={"configurable": {"thread_id": user_id}}  # 🧠 Thread-safe checkpointing
    )

    bot_reply = output["messages"][-1].content

    # Save to Firebase
    ref = db.reference(f"/chats/{user_id}")
    ref.push({"sender": "user", "message": user_msg})
    ref.push({"sender": "bot", "message": bot_reply})

    return jsonify({"reply": bot_reply})


@app.route("/history", methods=["GET"])
def get_history():
    user_id = request.args.get("user_id", "anon")
    ref = db.reference(f"/chats/{user_id}")
    messages = ref.order_by_key().limit_to_last(6).get()

    history = []
    if messages:
        for key in sorted(messages.keys()):
            history.append(messages[key])
    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True)

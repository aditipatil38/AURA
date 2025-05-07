from flask import Flask, render_template, request, jsonify
from firebase_config import db, init_firebase
from graph import app_graph
from langchain_core.messages import HumanMessage

app = Flask(__name__)
init_firebase()

@app.route("/")
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
        config={"configurable": {"thread_id": user_id}}
    )

    bot_reply = output["messages"][-1].content
    ref = db.reference(f"/chats/{user_id}")
    ref.push({"sender": "user", "message": user_msg})
    ref.push({"sender": "bot", "message": bot_reply})
    return jsonify({"reply": bot_reply})

@app.route("/history", methods=["GET"])
def get_history():
    user_id = request.args.get("user_id", "anon")
    ref = db.reference(f"/chats/{user_id}")
    messages = ref.order_by_key().limit_to_last(6).get()
    return jsonify([messages[k] for k in sorted(messages.keys())]) if messages else jsonify([])

if __name__ == "__main__":
    app.run(debug=True)



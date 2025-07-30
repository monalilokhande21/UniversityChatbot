# app.py

from flask import Flask, request, jsonify
from chatbot import get_qa_chain

app = Flask(__name__)
qa_chain = get_qa_chain()

@app.route("/")
def index():
    return "🎓 University Chatbot is Running!"

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "Please send a question."}), 400

    answer = qa_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

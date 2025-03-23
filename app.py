from flask import Flask, request, jsonify
from llm.llm_inference import get_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"

@app.route("/query", methods=["POST"])
def query():
    data=request.get_json()
    query = data["query"]
    response = get_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
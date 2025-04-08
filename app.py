from flask import Flask, request, jsonify, session
from llm.llm_inference import RAGPipeline
from auth.routes import auth_router
import jwt
from config import SECRET_KEY
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "some_random_secret")  
app.register_blueprint(auth_router, url_prefix="/api/auth")

def get_rag_pipeline():
    return RAGPipeline()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        
        try:
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            kwargs["user_payload"] = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except IndexError:
            return jsonify({"error": "Invalid Authorization header format"}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route("/api/health", methods=["GET"])
def index():
    return "Hello, World!"

@app.route("/api/query", methods=["POST"])
@token_required
def query(user_payload):
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data["query"]
    rag_pipeline = get_rag_pipeline()
    
    # Get or initialize chat history from session
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]
    
    try:
        # Pass session chat_history to RAGPipeline
        response = rag_pipeline.get_response(query, chat_history=chat_history)
        
       
        chat_history.append({"human": query, "ai": response})
        session["chat_history"] = chat_history[-10:]  # Limit to last 10 messages
        return jsonify({"response": response, "user_id": user_payload.get("user_id")}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
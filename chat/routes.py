from flask import Blueprint, request, jsonify, session
from llm.llm_inference import RAGPipeline
import jwt
from config import SECRET_KEY
from functools import wraps

# Define the chat Blueprint
chat_router = Blueprint("chat_router", __name__, url_prefix="/api/chat")

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

@chat_router.route("/query", methods=["POST"])
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
        response = rag_pipeline.get_response(query, chat_history=chat_history)
        chat_history.append({"human": query, "ai": response})
        session["chat_history"] = chat_history[-10:] 
        return jsonify({"response": response, "user_id": user_payload.get("user_id")}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500
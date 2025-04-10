from flask import Blueprint, request, jsonify, current_app
from llm.llm_inference import RAGPipeline
from chat.models import Chat, Message
import jwt
from config import SECRET_KEY
from functools import wraps
import uuid

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
    user_id = user_payload.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID missing from token"}), 401
    
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data["query"]
    chat_id = data.get("chat_id")
    chat_collection = current_app.config["CHAT_COLLECTION"]
    
    if not chat_id:
        chat = Chat(user_id=user_id)
        chat_dict = chat.dict()
        chat_collection.insert_one(chat_dict)
        chat_id = chat.chat_id
    else:
        chat_doc = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
        if not chat_doc:
            return jsonify({"error": "Chat session not found"}), 404
        chat = Chat(**chat_doc)
    
    chat_history = [Message(**msg) for msg in chat.messages]
    chat_history_dict = [msg.dict() for msg in chat_history]
    
    try:
        rag_pipeline = get_rag_pipeline()
        response = rag_pipeline.get_response(query, chat_history=chat_history_dict)
        
        new_message = Message(human=query, ai=response)
        chat_collection.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {"$push": {"messages": new_message.dict()}}
        )
        
        return jsonify({"response": response, "chat_id": chat_id, "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@chat_router.route("/query/<chat_id>", methods=["POST"])
@token_required
def query_with_chat_id(user_payload, chat_id):
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data["query"]
    user_id = user_payload.get("user_id")
    
    chat_collection = current_app.config["CHAT_COLLECTION"]
    chat_doc = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
    
    if not chat_doc:
        return jsonify({"error": "Chat session not found"}), 404
    
    chat = Chat(**chat_doc)
    chat_history = [Message(**msg) for msg in chat.messages]
    chat_history_dict = [msg.dict() for msg in chat_history]
    
    try:
        rag_pipeline = get_rag_pipeline()
        response = rag_pipeline.get_response(query, chat_history=chat_history_dict)
        
        new_message = Message(human=query, ai=response)
        chat_collection.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {"$push": {"messages": new_message.dict()}}
        )
        
        return jsonify({"response": response, "chat_id": chat_id, "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@chat_router.route("/history", methods=["GET"])
@token_required
def get_history(user_payload):
    user_id = user_payload.get("user_id")
    chat_id = request.args.get("chat_id")
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    
    chat_collection = current_app.config["CHAT_COLLECTION"]
    chat_doc = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
    
    if not chat_doc:
        return jsonify({"error": "Chat session not found"}), 404
    
    chat = Chat(**chat_doc)
    return jsonify({"chat_history": [msg.dict() for msg in chat.messages], "chat_id": chat_id, "user_id": user_id}), 200

@chat_router.route("/clear", methods=["POST"])
@token_required
def clear_history(user_payload):
    user_id = user_payload.get("user_id")
    chat_id = request.json.get("chat_id")
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    
    chat_collection = current_app.config["CHAT_COLLECTION"]
    result = chat_collection.update_one(
        {"chat_id": chat_id, "user_id": user_id},
        {"$set": {"messages": []}}
    )
    
    if result.matched_count == 0:
        return jsonify({"error": "Chat session not found"}), 404
    
    return jsonify({"message": "Chat history cleared", "chat_id": chat_id, "user_id": user_id}), 200
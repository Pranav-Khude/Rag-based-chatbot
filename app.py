from flask import Flask, request, jsonify
from llm.llm_inference import RAGPipeline
from auth.routes import auth_router
import jwt
from config import SECRET_KEY
from functools import wraps

rag_pipeline = RAGPipeline()

app = Flask(__name__)

app.register_blueprint(auth_router, url_prefix="/api/auth")

# Token verification decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401

        
        try:
            token = token.split(" ")[1]  # Extract token after "Bearer"
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
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
@token_required  # Protect this endpoint with authentication
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data["query"]
    response = rag_pipeline.get_response(query)
    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
from flask import Flask
from auth.routes import auth_router
from chat.routes import chat_router
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "some_random_secret")  # Still included for potential future use

# MongoDB setup for auth
mongo_uri = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
client = MongoClient(mongo_uri)
auth_db = client["auth"]
chat_db = client["chatbot_db"]

# Collections
user_collection = auth_db["user"]
chat_collection = chat_db["chat_histories"]

# Store collections in app config for access in routes
app.config["USER_COLLECTION"] = user_collection
app.config["CHAT_COLLECTION"] = chat_collection
# Register Blueprints
app.register_blueprint(auth_router, url_prefix="/api/auth")

# register chat router
app.register_blueprint(chat_router, url_prefix="/api/chat")

@app.route("/api/health", methods=["GET"])
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
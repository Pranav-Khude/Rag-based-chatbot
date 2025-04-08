from flask import Flask
from auth.routes import auth_router
from chat.routes import chat_router 
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "some_random_secret")  

# Register Blueprints
app.register_blueprint(auth_router, url_prefix="/api/auth")

# register chat router
app.register_blueprint(chat_router, url_prefix="/api/chat")

@app.route("/api/health", methods=["GET"])
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
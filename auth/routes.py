from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import jwt
from datetime import datetime, timedelta
from config import MONGO_DETAILS, SECRET_KEY
from auth.utils import send_verification_email, create_verification_token
from auth.models import UserSignup, UserLogin

auth_router = Blueprint('auth', __name__)
client = MongoClient(MONGO_DETAILS)
db = client["auth"]
user_collection = db["user"]

@auth_router.route('/signup', methods=['POST'])
def signup():
    try:
        data = UserSignup(**request.json).dict()
    except ValueError as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    existing_user = user_collection.find_one({"email": data['email']})
    if existing_user:
        return jsonify({"error": "Email already registered"}), 400

    hashed_pw = generate_password_hash(data['password'])
    user = {
        "email": data["email"],
        "username": data["username"],
        "password": hashed_pw,
        "verified": False
    }

    user_collection.insert_one(user)
    token = create_verification_token(data["email"], timedelta(hours=24)) 
    send_verification_email(data["email"], token)
    return jsonify({"msg": "User created, please verify email"}), 201

@auth_router.route('/verify_email', methods=['GET'])
def verify_email():
    token = request.args.get("token")
    if not token:
        return jsonify({"error": "Token is required"}), 400

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("sub")
        user = user_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 400
        if user.get("verified", False):
            return jsonify({"msg": "Email already verified"}), 200
        user_collection.update_one({"email": email}, {"$set": {"verified": True}})
        return jsonify({"msg": "Email verified successfully"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 400
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 400

@auth_router.route('/login', methods=['POST'])
def login():
    try:
        data = UserLogin(**request.json).dict()
    except ValueError as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    user = user_collection.find_one({"email": data['email']})
    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({"error": "Incorrect email or password"}), 400

    if not user.get("verified", False):
        return jsonify({"error": "Email not verified"}), 400

    token = jwt.encode({
        "email": user["email"],
        "exp": datetime.utcnow() + timedelta(days=1)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"]
    }), 200

@auth_router.route('/test_email', methods=['GET'])
def test_email():
    try:
        test_email = "test@example.com"  
        token = create_verification_token(test_email, timedelta(hours=24))
        send_verification_email(test_email, token)
        return jsonify({"message": "Test email sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to send test email: {str(e)}"}), 500
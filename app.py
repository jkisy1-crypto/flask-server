import os
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

FIREBASE_KEY_PATH = "/etc/secrets/firebase-key.json"

cred = credentials.Certificate(FIREBASE_KEY_PATH)

try:
    firebase_admin.initialize_app(cred, {
        "storageBucket": "dinoshuno-mos-app.firebasestorage.app"
    })
except ValueError:
    pass

@app.route("/")
def home():
    return "✅ DinoShuno Flask 서버 정상 작동"

@app.route("/train", methods=["POST"])
def train():
    return jsonify({"message": "train trigger ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

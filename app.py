from flask import Flask, jsonify
import os
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

firebase_app = None

def init_firebase():
    global firebase_app
    if firebase_app:
        return firebase_app

    key_path = os.getenv("FIREBASE_KEY_PATH", "/etc/secrets/firebase-key.json")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")

    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET env missing")

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"firebase key not found: {key_path}")

    cred = credentials.Certificate(key_path)
    firebase_app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    return firebase_app

def get_bucket():
    init_firebase()
    return storage.bucket()

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/firebase_test")
def firebase_test():
    bucket = get_bucket()

    # training_data 폴더에 뭐가 있나 20개만 확인
    blobs = bucket.list_blobs(prefix="training_data/", max_results=20)
    files = [b.name for b in blobs]

    return jsonify({
        "bucket": bucket.name,
        "count": len(files),
        "sample_files": files
    })

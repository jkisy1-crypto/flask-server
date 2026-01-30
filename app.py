import os
import io
import json
import time
import zipfile
import hashlib
from datetime import datetime, timezone

from flask import Flask, jsonify, request

import firebase_admin
from firebase_admin import credentials, storage


# =========================
# Flask
# =========================
app = Flask(__name__)


# =========================
# Firebase init (Secret Files)
# =========================
firebase_app = None

def init_firebase():
    """
    Render Secret Files:
      /etc/secrets/firebase-key.json
    Render Env:
      FIREBASE_KEY_PATH=/etc/secrets/firebase-key.json
      FIREBASE_STORAGE_BUCKET=dinoshuno-mos-app.firebasestorage.app
    """
    global firebase_app
    if firebase_app:
        return firebase_app

    key_path = os.getenv("FIREBASE_KEY_PATH", "/etc/secrets/firebase-key.json")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")

    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET env missing")

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Firebase key file not found: {key_path}")

    cred = credentials.Certificate(key_path)
    firebase_app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    return firebase_app

def get_bucket():
    init_firebase()
    return storage.bucket()


# =========================
# Utils
# =========================
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def put_json(bucket, path: str, obj: dict):
    blob = bucket.blob(path)
    blob.upload_from_string(
        json.dumps(obj, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

def list_blobs_safe(bucket, prefix: str, max_results: int = 20):
    blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
    return [b.name for b in blobs]

def download_blob_bytes(bucket, blob_path: str) -> bytes:
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()

def blob_exists(bucket, blob_path: str) -> bool:
    return bucket.blob(blob_path).exists()


# =========================
# Basic Routes
# =========================
@app.route("/", methods=["GET", "HEAD"])
def index():
    # 기존대로 메인페이지는 간단한 정상 문구
    return "✅ DinoShuno Flask 서버 정상 작동", 200

@app.route("/health", methods=["GET", "HEAD"])
def health():
    # UptimeRobot은 여기를 치게 권장
    return jsonify({
        "status": "ok",
        "time": now_utc_iso()
    }), 200

@app.route("/firebase_test", methods=["GET"])
def firebase_test():
    """
    Firebase Storage 연결 확인.
    training_data/ 아래 파일을 일부 나열.
    """
    try:
        bucket = get_bucket()
        files = list_blobs_safe(bucket, prefix="training_data/", max_results=30)
        return jsonify({
            "ok": True,
            "bucket": bucket.name,
            "count": len(files),
            "sample_files": files
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# Lock (prevent concurrent train)
# =========================
LOCK_BLOB = "training_data/train.lock"

def acquire_lock(bucket, ttl_seconds: int = 600) -> bool:
    """
    Firebase Storage에 lock 파일을 두고 동시 실행 방지.
    ttl_seconds 지나면 stale lock으로 간주하고 덮어씀.
    """
    try:
        blob = bucket.blob(LOCK_BLOB)
        if blob.exists():
            data = blob.download_as_text()
            try:
                lock_info = json.loads(data)
                ts = lock_info.get("ts", 0)
                if time.time() - ts < ttl_seconds:
                    return False
            except:
                # 파싱 실패면 stale로 간주
                pass

        put_json(bucket, LOCK_BLOB, {"ts": time.time(), "time": now_utc_iso()})
        return True
    except:
        return False

def release_lock(bucket):
    try:
        blob = bucket.blob(LOCK_BLOB)
        if blob.exists():
            blob.delete()
    except:
        pass


# =========================
# Create data.zip (images+labels)
# =========================
def build_training_zip_from_storage(bucket,
                                   images_prefix="training_data/images/",
                                   labels_prefix="training_data/labels/",
                                   max_files=3000) -> bytes:
    """
    Render에서 학습은 하지 않는다.
    대신 images/labels를 모아서 data.zip 생성해서 Firebase에 업로드.
    """
    images = list(bucket.list_blobs(prefix=images_prefix, max_results=max_files))
    labels = list(bucket.list_blobs(prefix=labels_prefix, max_results=max_files))

    # 파일명->blob 매핑
    label_map = {}
    for b in labels:
        name = b.name.split("/")[-1]
        label_map[name] = b

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        added = 0
        skipped = 0

        for img_blob in images:
            img_name = img_blob.name.split("/")[-1]
            if not img_name:
                continue

            base = os.path.splitext(img_name)[0]
            label_name = base + ".txt"
            if label_name not in label_map:
                skipped += 1
                continue

            # 다운로드
            img_bytes = img_blob.download_as_bytes()
            label_bytes = label_map[label_name].download_as_bytes()

            # zip 내부 경로 (dataset 구조는 네 훈련코드가 맞춰서 사용)
            z.writestr(f"images/{img_name}", img_bytes)
            z.writestr(f"labels/{label_name}", label_bytes)
            added += 1

        # 메타정보
        meta = {
            "created_at": now_utc_iso(),
            "images_prefix": images_prefix,
            "labels_prefix": labels_prefix,
            "added_pairs": added,
            "skipped_no_label": skipped
        }
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    mem.seek(0)
    return mem.read()


# =========================
# /train : job trigger
# =========================
@app.route("/train", methods=["POST", "GET"])
def train():
    """
    Android -> https://www.dinoshuno.com/train 호출

    여기서는 "학습 실행"이 아니라:
      1) 동시실행 방지 lock
      2) training_data/images + labels -> data.zip 생성
      3) storage training_data/data.zip 업로드
      4) training_data/train_request.json 업로드 (데스크탑 GPU 에이전트가 이걸 보고 학습 수행)
    """
    bucket = None
    try:
        bucket = get_bucket()

        if not acquire_lock(bucket):
            return jsonify({"ok": False, "status": "locked"}), 429

        # zip 생성
        zip_bytes = build_training_zip_from_storage(bucket)

        # 업로드
        zip_path = "training_data/data.zip"
        blob = bucket.blob(zip_path)
        blob.upload_from_string(zip_bytes, content_type="application/zip")

        zip_hash = sha256_bytes(zip_bytes)

        # train 요청 기록 (GPU 데스크탑 에이전트가 이거 보고 학습)
        req = {
            "requested_at": now_utc_iso(),
            "zip_path": zip_path,
            "zip_sha256": zip_hash,
            "bucket": bucket.name,
            "client_ip": request.headers.get("X-Forwarded-For", request.remote_addr),
            "note": "Render created data.zip only. Actual training should run on desktop GPU agent."
        }
        put_json(bucket, "training_data/train_request.json", req)

        return jsonify({
            "ok": True,
            "message": "data.zip created and train_request.json updated",
            "zip_path": zip_path,
            "zip_sha256": zip_hash
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        if bucket:
            release_lock(bucket)


# =========================
# Model metadata endpoint (optional)
# =========================
@app.route("/model_metadata", methods=["GET"])
def model_metadata():
    """
    Android 업데이트 매니저가 참고할 메타데이터.
    (추후 trained_models/metadata.json로 확장)
    """
    try:
        bucket = get_bucket()
        meta_path = "trained_models/metadata.json"
        if not blob_exists(bucket, meta_path):
            return jsonify({"ok": False, "error": "metadata.json not found"}), 404

        data = download_blob_bytes(bucket, meta_path)
        return app.response_class(data, mimetype="application/json")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

import os
import json
import time
import hashlib
import zipfile
from datetime import datetime, timezone

from flask import Flask, jsonify
from flask import request, jsonify

import firebase_admin
from firebase_admin import credentials, storage


# =========================
# Config
# =========================
DEFAULT_BUCKET = os.getenv("FIREBASE_BUCKET", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# 보안/안정 설정
DISABLE_TRAIN = os.getenv("DISABLE_TRAIN", "0") == "1"   # 응급 차단용
TRAIN_TOKEN = os.getenv("TRAIN_TOKEN", "").strip()       # 반드시 설정 권장
TRAIN_MIN_INTERVAL = int(os.getenv("TRAIN_MIN_INTERVAL", "600"))  # 최소 호출 간격(초) 기본 10분

# Storage 경로
TRAINING_DATA_PREFIX = os.getenv("TRAINING_DATA_PREFIX", "training_data")
TRAIN_REQUEST_PATH = os.getenv("TRAIN_REQUEST_PATH", f"{TRAINING_DATA_PREFIX}/train_request.json")

# zip 파일은 고유 경로로 저장
ZIPS_PREFIX = os.getenv("ZIPS_PREFIX", f"{TRAINING_DATA_PREFIX}/zips")

# zip 만들 대상 폴더
IMAGES_PREFIX = os.getenv("IMAGES_PREFIX", f"{TRAINING_DATA_PREFIX}/images")
LABELS_PREFIX = os.getenv("LABELS_PREFIX", f"{TRAINING_DATA_PREFIX}/labels")

app = Flask(__name__)

firebase_inited = False
last_train_ts = 0


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def init_firebase():
    global firebase_inited
    if firebase_inited:
        return

    if not DEFAULT_BUCKET:
        raise RuntimeError("FIREBASE_BUCKET env is missing")

    # 1) GOOGLE_APPLICATION_CREDENTIALS가 file path면 그걸 사용
    if GOOGLE_APPLICATION_CREDENTIALS:
        if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
        cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
        firebase_admin.initialize_app(cred, {"storageBucket": DEFAULT_BUCKET})
        firebase_inited = True
        return

    # 2) fallback: Secret File 경로 고정
    secret_path = "/etc/secrets/firebase-key.json"
    if os.path.exists(secret_path):
        cred = credentials.Certificate(secret_path)
        firebase_admin.initialize_app(cred, {"storageBucket": DEFAULT_BUCKET})
        firebase_inited = True
        return

    raise RuntimeError("Firebase key not found. Set GOOGLE_APPLICATION_CREDENTIALS or upload secret file.")


def get_bucket():
    init_firebase()
    return storage.bucket()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_train_token():
    """
    /train은 반드시 토큰 있어야 한다.
    token 없으면 폭격 방지 불가.
    """
    if not TRAIN_TOKEN:
        return False  # 토큰 없으면 막아버림

    token = request.args.get("token") or request.headers.get("X-TRAIN-TOKEN")
    return token == TRAIN_TOKEN


def rate_limit_train():
    global last_train_ts
    now = time.time()
    if now - last_train_ts < TRAIN_MIN_INTERVAL:
        return False, int(TRAIN_MIN_INTERVAL - (now - last_train_ts))
    last_train_ts = now
    return True, 0


def list_blobs(bucket, prefix: str):
    return list(bucket.list_blobs(prefix=prefix))


def build_zip_from_storage(bucket, images_prefix: str, labels_prefix: str, out_zip_local: str):
    os.makedirs(os.path.dirname(out_zip_local), exist_ok=True)

    image_blobs = list_blobs(bucket, images_prefix)
    label_blobs = list_blobs(bucket, labels_prefix)

    # Render free 안전장치
    max_files = int(os.getenv("ZIP_MAX_FILES", "5000"))
    total = len(image_blobs) + len(label_blobs)
    if total > max_files:
        raise RuntimeError(f"Too many files for zip on Render: {total} > {max_files}")

    with zipfile.ZipFile(out_zip_local, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for b in image_blobs:
            if b.name.endswith("/"):
                continue
            data = b.download_as_bytes()
            arc = b.name.replace(images_prefix + "/", "images/")
            z.writestr(arc, data)

        for b in label_blobs:
            if b.name.endswith("/"):
                continue
            data = b.download_as_bytes()
            arc = b.name.replace(labels_prefix + "/", "labels/")
            z.writestr(arc, data)


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return "DinoShuno Flask 서버 정상 작동", 200


@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": utc_now_iso()}), 200


@app.get("/firebase_test")
def firebase_test():
    try:
        bucket = get_bucket()
        blobs = bucket.list_blobs(prefix=TRAINING_DATA_PREFIX)
        sample = []
        count = 0
        for b in blobs:
            count += 1
            if len(sample) < 10:
                sample.append(b.name)
        return jsonify({"ok": True, "bucket": DEFAULT_BUCKET, "count": count, "sample_files": sample}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/train", methods=["POST"])   # ✅ POST만 허용 (GET 차단)
def train():
    """
    Render에서는:
    - 학습 X
    - data zip 생성만
    - train_request.json 갱신만
    """
    try:
        if DISABLE_TRAIN:
            return jsonify({"ok": False, "error": "train disabled temporarily"}), 403

        # ✅ token 필수
        if not require_train_token():
            return jsonify({"ok": False, "error": "unauthorized (token required)"}), 401

        # ✅ 레이트리밋
        ok, wait_sec = rate_limit_train()
        if not ok:
            return jsonify({"ok": False, "error": f"rate limited, wait {wait_sec}s"}), 429

        # ✅ 폭격 범인 로그
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua = request.headers.get("User-Agent", "")
        app.logger.warning(f"[TRAIN] called ip={ip} ua={ua}")

        bucket = get_bucket()

        # --- 고유 zip 파일명 생성 ---
        stamp = utc_stamp()
        zip_path = f"{ZIPS_PREFIX}/data_{stamp}.zip"

        work_dir = "/tmp/mosquito_train"
        os.makedirs(work_dir, exist_ok=True)
        zip_local = os.path.join(work_dir, f"data_{stamp}.zip")

        # --- zip 생성 ---
        build_zip_from_storage(bucket, IMAGES_PREFIX, LABELS_PREFIX, zip_local)

        # --- zip 업로드 ---
        blob = bucket.blob(zip_path)
        blob.upload_from_filename(zip_local, content_type="application/zip")

        # --- sha256 계산 ---
        zip_bytes = bucket.blob(zip_path).download_as_bytes()
        zip_hash = sha256_bytes(zip_bytes)

        # --- train_request.json 업데이트 ---
        req = {
            "ok": True,
            "updated_at": utc_now_iso(),
            "zip_path": zip_path,
            "zip_sha256": zip_hash,
            "images_prefix": IMAGES_PREFIX,
            "labels_prefix": LABELS_PREFIX,
            "note": "Render created zip only. Desktop GPU trainer should download zip_path and train.",
            "status": "ready_for_training"
        }

        bucket.blob(TRAIN_REQUEST_PATH).upload_from_string(
            json.dumps(req, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

        return jsonify({
            "ok": True,
            "message": "zip created and train_request.json updated",
            "zip_path": zip_path,
            "zip_sha256": zip_hash,
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

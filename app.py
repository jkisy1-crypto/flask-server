import os
import json
import time
import hashlib
import zipfile
from datetime import datetime, timezone

from flask import Flask, jsonify, request

import firebase_admin
from firebase_admin import credentials, storage


# =========================
# Config
# =========================
DEFAULT_BUCKET = os.getenv("FIREBASE_BUCKET", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# 보안/안정 설정
DISABLE_TRAIN = os.getenv("DISABLE_TRAIN", "0") == "1"     # 응급 차단용
TRAIN_TOKEN = os.getenv("TRAIN_TOKEN", "").strip()         # 토큰 없으면 제한적으로라도 방어
TRAIN_MIN_INTERVAL = int(os.getenv("TRAIN_MIN_INTERVAL", "30"))  # 최소 호출 간격(초)

# Storage 경로
TRAINING_DATA_PREFIX = os.getenv("TRAINING_DATA_PREFIX", "training_data")
TRAIN_REQUEST_PATH = os.getenv("TRAIN_REQUEST_PATH", f"{TRAINING_DATA_PREFIX}/train_request.json")
ZIP_PATH = os.getenv("ZIP_PATH", f"{TRAINING_DATA_PREFIX}/data.zip")

# zip 만들 대상 폴더
IMAGES_PREFIX = os.getenv("IMAGES_PREFIX", f"{TRAINING_DATA_PREFIX}/images")
LABELS_PREFIX = os.getenv("LABELS_PREFIX", f"{TRAINING_DATA_PREFIX}/labels")


app = Flask(__name__)

firebase_inited = False
last_train_ts = 0


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_firebase():
    """
    - Render Secret Files: /etc/secrets/firebase-key.json
    - env: GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/firebase-key.json
    """
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
    토큰이 설정되어 있으면 반드시 token 일치해야 함.
    (없으면 완전 오픈인데, 최소 레이트리밋은 적용)
    """
    if not TRAIN_TOKEN:
        return True

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
    """
    Render에서 너무 무거운 학습을 하지 않기 위해:
    - images, labels 파일을 zip으로만 묶는다.
    - 이 과정도 최대한 가볍게.
    """
    os.makedirs(os.path.dirname(out_zip_local), exist_ok=True)

    image_blobs = list_blobs(bucket, images_prefix)
    label_blobs = list_blobs(bucket, labels_prefix)

    # 최소 방어: 데이터가 너무 많으면 Render free에서 죽을 수 있음
    max_files = int(os.getenv("ZIP_MAX_FILES", "5000"))
    if len(image_blobs) + len(label_blobs) > max_files:
        raise RuntimeError(f"Too many files for zip on Render: {len(image_blobs)+len(label_blobs)} > {max_files}")

    with zipfile.ZipFile(out_zip_local, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # images
        for b in image_blobs:
            if b.name.endswith("/"):
                continue
            data = b.download_as_bytes()
            arc = b.name.replace(images_prefix + "/", "images/")
            z.writestr(arc, data)

        # labels
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
    # 가장 가벼운 엔드포인트
    return jsonify({"status": "ok", "time": utc_now_iso()}), 200


@app.get("/firebase_test")
def firebase_test():
    """
    bucket 접속 가능한지 + 폴더 구조 확인
    """
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


@app.route("/train", methods=["GET", "POST"])
def train():
    """
    중요:
    - Render에서는 학습 절대 하지 않는다.
    - data.zip 생성 + train_request.json 업데이트만 한다.
    - desktop trainer가 train_request.json 보고 GPU 학습함.
    """
    try:
        # --- 0) 즉시 차단 옵션 ---
        if DISABLE_TRAIN:
            return jsonify({"ok": False, "error": "train disabled temporarily"}), 403

        # --- 1) 토큰 체크 ---
        if not require_train_token():
            return jsonify({"ok": False, "error": "unauthorized (token required)"}), 401

        # --- 2) 레이트 리밋 ---
        ok, wait_sec = rate_limit_train()
        if not ok:
            return jsonify({"ok": False, "error": f"rate limited, wait {wait_sec}s"}), 429

        # --- 3) 호출자 정보 로그 (폭격 범인 추적용) ---
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua = request.headers.get("User-Agent", "")
        app.logger.warning(f"[TRAIN] called ip={ip} ua={ua}")

        bucket = get_bucket()

        # --- 4) zip 생성 ---
        work_dir = "/tmp/mosquito_train"
        os.makedirs(work_dir, exist_ok=True)

        zip_local = os.path.join(work_dir, "data.zip")
        build_zip_from_storage(bucket, IMAGES_PREFIX, LABELS_PREFIX, zip_local)

        # --- 5) zip 업로드 ---
        blob = bucket.blob(ZIP_PATH)
        blob.upload_from_filename(zip_local, content_type="application/zip")

        # upload 후 다시 bytes로 가져와 hash 계산(정확성 위해)
        zip_bytes = bucket.blob(ZIP_PATH).download_as_bytes()
        zip_hash = sha256_bytes(zip_bytes)

        # --- 6) train_request.json 업데이트 ---
        req = {
            "ok": True,
            "updated_at": utc_now_iso(),
            "zip_path": ZIP_PATH,
            "zip_sha256": zip_hash,
            "note": "Render created data.zip only. Training runs on Desktop GPU.",
        }
        bucket.blob(TRAIN_REQUEST_PATH).upload_from_string(
            json.dumps(req, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

        return jsonify({
            "ok": True,
            "message": "data.zip created and train_request.json updated",
            "zip_path": ZIP_PATH,
            "zip_sha256": zip_hash,
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # local debug only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

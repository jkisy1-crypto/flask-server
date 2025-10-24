# app.py
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, storage
import torch
import torch.nn as nn
import torchvision.models as models
import zipfile, os, requests

# -------------------------------
# Flask 앱 초기화
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Firebase 서비스 계정 키 로드
# Render에서는 Secret Files에 firebase-key.json 업로드 필요
# -------------------------------
cred = credentials.Certificate("firebase-key.json")

# -------------------------------
# Firebase 초기화 (이미 초기화된 경우 예외 방지)
# -------------------------------
try:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'dinoshuno-mos-app.firebasestorage.app'
    })
except ValueError:
    pass

# -------------------------------
# 루트 페이지 (서버 정상 작동 확인용)
# -------------------------------
@app.route('/')
def home():
    return "✅ Flask 서버 정상 작동 중입니다! (DinoShuno MOS 연결 완료)"

# -------------------------------
# 학습 API (Firebase Function에서 호출)
# -------------------------------
@app.route('/train', methods=['POST'])
def train():
    print("🔥 Firebase Function에서 학습 요청 수신!")

    # Storage에서 학습 zip 파일 다운로드
    try:
        bucket = storage.bucket()
        blob = bucket.blob('training_data/data.zip')
        local_zip_path = 'data.zip'
        blob.download_to_filename(local_zip_path)
        print("📦 학습 데이터 다운로드 완료")
    except Exception as e:
        print(f"❌ 학습 데이터 다운로드 실패: {e}")
        return jsonify({"error": str(e)}), 500

    # zip 압축 해제
    try:
        extract_dir = 'training_data'
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("📂 학습 데이터 압축 해제 완료")
    except Exception as e:
        print(f"❌ 압축 해제 실패: {e}")
        return jsonify({"error": str(e)}), 500

    # 샘플 모델 학습
    try:
        print("🤖 모델 학습 시작...")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 클래스 예시

        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print("✅ 모델 Forward Pass 완료")

        # 학습 완료 후 모델 저장
        os.makedirs("trained_models", exist_ok=True)
        model_path = "trained_models/model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"💾 모델 저장 완료: {model_path}")

        # Firebase Storage 업로드
        upload_blob = bucket.blob('trained_models/model.pt')
        upload_blob.upload_from_filename(model_path)
        print("☁️ 모델 Firebase Storage 업로드 완료")

    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

    # 성공 응답
    return jsonify({"message": "🎯 학습 완료 및 모델 업로드 성공!"}), 200


# -------------------------------
# Render 서버 실행 설정
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

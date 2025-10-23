from flask import Flask, request
import firebase_admin
from firebase_admin import credentials, storage
import os
import zipfile
import torch  # 예시용 (PyTorch 기반)
import torchvision.models as models
import torch.nn as nn

# Flask 앱 초기화
app = Flask(__name__)

# ✅ Firebase 서비스 계정 키 로드
# Render에서는 firebase-key.json을 Secret Files에 업로드해야 함
cred = credentials.Certificate("firebase-key.json")

# ✅ Firebase 초기화 (이미 초기화된 경우 예외 방지)
try:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'dinoshuno-mos-app.appspot.com'
    })
except ValueError:
    pass


# ✅ 루트 페이지 (상태 확인용)
@app.route('/')
def home():
    return "✅ Flask 서버 정상 작동 중입니다! (DinoShuno MOS 연결 완료)"


# ✅ 학습 요청 API (Firebase Functions가 POST 요청 보냄)
@app.route('/train', methods=['POST'])
def train():
    print("📩 Firebase Function에서 학습 요청 수신")

    try:
        # 1️⃣ 학습용 zip 파일 다운로드
        bucket = storage.bucket()
        blob = bucket.blob('training_data/data.zip')
        local_zip_path = 'data.zip'
        blob.download_to_filename(local_zip_path)
        print("📦 학습 데이터 다운로드 완료")

        # 2️⃣ 압축 해제
        extract_dir = 'training_data'
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("📂 학습 데이터 압축 해제 완료")

        # 3️⃣ 학습 실행 (샘플 코드)
        print("🧠 모델 학습 시작...")
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 클래스 수 2 예시

        # 예시 학습 루프 (실제 프로젝트에서는 데이터셋으로 교체)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print("✅ 모델 Forward Pass 완료 (샘플 학습 실행됨)")

        # 4️⃣ 모델 저장
        os.makedirs("trained_models", exist_ok=True)
        model_path = "trained_models/model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"💾 모델 저장 완료: {model_path}")

        # 5️⃣ 학습 완료 후 Firebase Storage에 업로드
        upload_blob = bucket.blob('trained_models/model.pt')
        upload_blob.upload_from_filename(model_path)
        print("🚀 모델 Firebase Storage 업로드 완료")

        return "🎯 학습 완료 및 모델 업로드 성공!", 200

    except Exception as e:
        print(f"❌ 학습 실패 또는 업로드 실패: {e}")
        return f"학습 실패: {e}", 500


# ✅ Render 배포 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

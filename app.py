from flask import Flask, request
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

# ✅ Firebase 서비스 계정 키 로드
# Render에서는 firebase-key.json을 Secret Files로 올려야 함
cred = credentials.Certificate("firebase-key.json")

# ✅ Firebase 초기화 (이미 초기화된 경우 예외 방지)
try:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'dinoshuno-mos-app.appspot.com'
    })
except ValueError:
    pass

# ✅ 루트 페이지 (서버 상태 확인용)
@app.route('/')
def home():
    return "✅ Flask 서버 정상 작동 중입니다! (DinoShuno MOS 연결 완료)"

# ✅ 학습 API (Firebase Functions가 호출)
@app.route('/train', methods=['POST'])
def train():
    print("🔥 Firebase Function에서 학습 요청 받음")

    # ① Storage에서 학습용 zip 다운로드
    try:
        bucket = storage.bucket()
        blob = bucket.blob('training_data/data.zip')
        blob.download_to_filename('data.zip')
        print("✅ 학습 데이터 다운로드 완료")
    except Exception as e:
        print("❌ 학습 데이터 다운로드 실패:", e)
        return f"학습 데이터 다운로드 실패: {e}", 500

    # ② 예시 학습 로직 (임시)
    print("🧠 모델 학습 중... (샘플 코드 실행)")
    # 여기서 실제 학습 코드를 실행하면 됨 (예: PyTorch, TensorFlow 등)

    # ③ 학습 완료 후 모델 업로드
    try:
        trained_model = 'trained_models/model.pt'
        blob = bucket.blob(trained_model)
        blob.upload_from_filename(trained_model)
        print("🚀 모델 업로드 완료")
    except Exception as e:
        print("❌ 모델 업로드 실패:", e)
        return f"모델 업로드 실패: {e}", 500

    return "🎯 학습 완료 및 모델 업로드 성공!", 200


# ✅ Render 배포용 서버 구동
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

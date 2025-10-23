from flask import Flask, request
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

# Firebase 연결
cred = credentials.Certificate("/etc/secrets/firebase-key.json")  # 수정된 부분
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dinoshuno-mos-app.appspot.com'
})

@app.route('/train', methods=['POST'])
def train():
    print("📩 Firebase Function에서 학습 요청 받음")

    # ① Storage에서 학습용 데이터 불러오기
    bucket = storage.bucket()
    blob = bucket.blob('training_data/data.zip')
    blob.download_to_filename('data.zip')
    print("✅ 학습 데이터 다운로드 완료")

    # ② 모델 학습 (예시)
    print("🧠 모델 학습 중...")
    # 실제 학습 코드 들어갈 위치

    # ③ 학습 완료된 모델 업로드
    trained_model = 'trained_model/model.pt'
    blob = bucket.blob(trained_model)
    blob.upload_from_filename(trained_model)
    print("✅ 모델 업로드 완료")

    return "학습 완료 및 모델 업로드 성공", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

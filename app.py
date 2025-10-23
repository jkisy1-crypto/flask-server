from flask import Flask, request
import os
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

# Firebase 연결
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dinoshuno-mos-app.appspot.com'
})

@app.route('/train', methods=['POST'])
def train():
    print("🔥 Firebase Function에서 학습 요청 받음")
    # 1️⃣ 업로드된 데이터 로드 (예: 이미지 파일)
    # 2️⃣ 모델 학습 실행
    # 3️⃣ 결과 모델 파일을 Firebase Storage에 업로드
    bucket = storage.bucket()
    blob = bucket.blob('trained_model/model.pt')
    blob.upload_from_filename('model.pt')
    return "학습 완료 및 모델 업로드 성공", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

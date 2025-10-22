from flask import Flask, request

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    print("🔥 Firebase Function에서 학습 요청 받음")
    # 여기에 학습 로직 실행 코드 추가
    return "학습 시작", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

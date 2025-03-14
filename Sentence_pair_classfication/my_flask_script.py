import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify, send_file

class TrainArgs:
    def __init__(self):
        self.pretrained_model_name = "beomi/kcbert-base"
        self.batch_size = 32 if torch.cuda.is_available() else 4
        self.learning_rate = 5e-5
        self.max_seq_length = 64
        self.epochs = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "./bert_nli_model"

# args 변수를 전역 변수로 설정 (Flask에서도 접근 가능하도록)
args = TrainArgs()

# 모델과 토크나이저 로드
MODEL_PATH = "./bert_nli_model"  # 학습된 모델이 저장된 경로
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def inference_fn(premise, hypothesis):
    inputs = tokenizer([(premise, hypothesis)], max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prob = F.softmax(outputs.logits, dim=1)
        predicted_index = torch.argmax(prob, dim=1).item()

        LABEL_MAPPING = {
            1: "참 (entailment)",       # index 1이 "참"
            0: "거짓 (contradiction)",  # index 0이 "거짓"
            2: "중립 (neutral)"         # index 2가 "중립"
        }

        pred = LABEL_MAPPING[predicted_index]

    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'prediction': pred,
        'entailment_data': f"참 {round(prob[0][1].item(), 2)}",  # index 1이 entailment
        'contradiction_data': f"거짓 {round(prob[0][0].item(), 2)}",  # index 0이 contradiction
        'neutral_data': f"중립 {round(prob[0][2].item(), 2)}",  # index 2가 neutral
        'entailment_width': f"{prob[0][1].item() * 100}%",
        'contradiction_width': f"{prob[0][0].item() * 100}%",
        'neutral_width': f"{prob[0][2].item() * 100}%",
    }


app = Flask(__name__)

@app.route('/')
def serve_index():
    return send_file("index.html")  # index.html을 반환

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    premise = data.get("premise")
    hypothesis = data.get("hypothesis")

    if not premise or not hypothesis:
        return jsonify({"error": "premise and hypothesis are required"}), 400

    result = inference_fn(premise, hypothesis)  # 인퍼런스 실행
    return jsonify(result)  # JSON 응답 반환

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Flask 서버 실행

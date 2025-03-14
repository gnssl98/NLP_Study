import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 모델과 토크나이저 로드
MODEL_PATH = "./bert_nli_model"  # 학습된 모델이 저장된 경로
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 테스트 데이터셋 로드 (KLUE NLI 데이터셋 사용)
dataset = load_dataset("klue", "nli")
test_dataset = dataset["validation"]  # 테스트용 데이터셋

# KLUE NLI의 실제 라벨 순서 적용 (올바르게 수정)
LABEL_MAPPING = {
    "contradiction": 0,  # 거짓
    "entailment": 1,     # 참
    "neutral": 2         # 중립
}

# 모델 성능 평가 함수
def evaluate_model(model, tokenizer, test_dataset):
    predictions = []
    true_labels = []

    for example in test_dataset:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label_str = example["label"]  # KLUE NLI에서는 문자열 라벨이 제공됨
        true_label = LABEL_MAPPING[true_label_str]  # 문자열 라벨을 숫자로 변환

        # 입력 데이터 토큰화
        inputs = tokenizer([(premise, hypothesis)], max_length=64, padding="max_length", truncation=True, return_tensors="pt")

        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()

        predictions.append(predicted_label)
        true_labels.append(true_label)

    #  성능 평가 지표 계산
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    print("\n 모델 성능 평가 결과 ")
    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model(model, tokenizer, test_dataset)

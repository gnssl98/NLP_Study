import requests

url = "http://localhost:5001/predict"
data = {
    "premise": "나는 사과를 좋아해",
    "hypothesis": "나는 과일을 좋아해"
}

response = requests.post(url, json=data)
print(response.json())

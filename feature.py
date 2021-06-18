import requests
import json

url = 'https://api.aicloud.sbercloud.ru/public/v1/public_inference/gpt3/predict'
data = {"text": "Гороскоп ОВЕН на неделю:"}

response = requests.post("https://api.aicloud.sbercloud.ru/public/v1/public_inference/gpt3/predict", verify=True, json=data)
d = json.loads(response.text)
import requests
import json
import pymorphy2
import nltk
from rutimeparser import parse
import datetime
from dateutil import parser


class HoroscopeParser:
    keywords = ["гороскоп", "предсказание", "судьба", "астрологический"]
    url = 'https://api.aicloud.sbercloud.ru/public/v1/public_inference/gpt3/predict'
    morph = pymorphy2.MorphAnalyzer()
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    horo_signs = {'овен': 'овен', 'телец': 'телец',
                  'близнец': 'близнецы', 'рак': 'рак',
                  'лев': 'лев', 'дева': 'дева',
                  'вес': 'весы', 'весы': 'весы',
                  'скорпион': 'скорпион', 'стрелец': 'стрелец',
                  'козерог': 'козерог', 'водолей': 'водолей',
                  'рыбы': 'рыбы', 'рыба': 'рыбы'}
    tags = ['B-TIME', 'I-TIME', 'B-DATE', 'I-DATE']

    def process_date(self, message, ner_model):
        date = parse(message)

        if date is None:
            date = ''
            result = ner_model([message])
            for idx in [i for i, el in enumerate(result[1][0]) if el in self.tags]:
                date += result[0][0][idx]
            date = '' if not date else parser.parse(date).date()

        if not date or date < datetime.datetime.now().date():
            date = datetime.datetime.now().date()
        return date

    def process_sign(self, message):
        result = self._tokenize_text(message)
        sign = [word for word in result if word in self.horo_signs]
        if not sign:
            return None
        return self.horo_signs[sign[0]]

    def get_horo(self, date, horo_sign):
        data = {"text": f"Гороскоп {horo_sign} на {date}:"}
        response = requests.post(self.url, verify=True, json=data)
        horo = json.loads(response.text)['predictions']
        return horo

    def _tokenize_text(self, text):
        words = self.tokenizer.tokenize(text)
        result = []

        for word in words:
            p = self.morph.parse(word)[0]
            result.append(p.normal_form)
        return result
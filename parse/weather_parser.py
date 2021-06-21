from parse.command_parser import CommandParser
import requests
import re
import datetime
import json
from rutimeparser import parse

from smth import day_dict_proc

from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, NewsSyntaxParser
from natasha.morph.tagger import NewsMorphTagger

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
ner_tagger = NewsNERTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

str2num = {
    'один': '1',
    'два': '2',
    'три': '3',
    'четыре': '4',
    'пять': '5',
    'шесть': '6',
    'семь': '7',
    }

main_cities ={
    'Санкт-Петербург': ['спб', 'питер', 'ленинград', 'петроград', 'петербург'],
    'Москва': ['мск', 'москва'],
    'Ростов-на-Дону': ['ростов', 'ростов-на-дон'],
    'Екатеринбург': ['екб', 'екатеринбург'],
    'Нарьян-Мар': ['нм', 'нарьян-мар']
}

class WeatherParser(CommandParser):

    def __init__(self):
        self.keywords = ["прогноз", "температура", "погода", "градус"]
        self.date = None
        self.period = None
        self.city = None
        self.lat = None
        self.lon = None

    def get_date(self, message, ner_model):
        copy_message = message[:]
        for k, v in str2num.items():
            if k in message:
                copy_message = copy_message.replace(k, v)

        if 'на' in copy_message:
            split_message = copy_message.split()
            idx = split_message.index('на')
            period_text = ''
            if split_message[idx + 1].isdigit():
                period_text = ' '.join(split_message[idx: idx + 3])
            else:
                if split_message[idx: idx + 2] not in ['сегодня', 'завтра']:
                    period_text = ' '.join(split_message[idx: idx + 2])

            self.get_period(period_text, ner_model)
            copy_message = copy_message.replace(period_text, "")

        today = datetime.datetime.now().date()
        date = parse(copy_message)
        if date is not None:
            try:
                date = date.date()
            except Exception as e:
                date = date
            if date < today:  # пн - вс, недел -- могут в обратную сторону распарсится
                date = date + datetime.timedelta(days=7)
            self.date = date
            # return self.date

        else:
            res = re.search(r'\d?\d\.\d{2}(?:\.\d{4}|\.\d{2})?', copy_message)
            if res is not None:
                res = res.group(0)
                if len(res) in [4, 5]:
                    date = datetime.datetime.strptime(res, '%d.%m').date()
                    cur_year = datetime.datetime.now().year
                    if date.year < cur_year:
                        date = date.replace(year=cur_year)
                elif len(res) in [7, 8]:
                    date = datetime.datetime.strptime(res, '%d.%m.%y').date()
                else:
                    date = datetime.datetime.strptime(res, '%d.%m.%Y').date()

            if date:
                cur_date = datetime.datetime.now().date()
                if date < cur_date:
                    date = cur_date
                self.date = date
                # return self.date
            else:
                self.date = datetime.datetime.now().date()
                # return self.date

    def get_period(self, message, ner_model):
        res = []
        tags = ['B-DATE', 'I-DATE', 'B-TIME', 'I-TIME']
        ner_res = ner_model([message])
        for i, t in enumerate(ner_res[1][0]):
            if t in tags:
                res.append(ner_res[0][0][i])

        if len(res) == 2:
            if res[0] in str2num.keys():
                self.period = int(str2num[res[0]])
                # return int(str2num[res[0]])
            elif res[0].isdigit() and (res[0] in str2num.values()):
                self.period = min(7, int(res[0]))
                # return min(7, int(res[0]))
            else:
                self.period = 7
                # return 7
        elif 'день' in message:
            self.period = 1
            # return 1
        elif 'недел' in message:
            self.period = 7
            # return 7
        else:
            self.period = None
            # return None

    def get_city(self, message):
        # сначала по аббревеатурам
        for k, v in main_cities.items():
            for c in v:
                if c in message.lower():
                    self.city = k
                    return # return k

        # потом всё остальное
        # natasha
        document = Doc(message)
        document.segment(segmenter)
        document.tag_morph(morph_tagger)
        for token in document.tokens:
            token.lemmatize(morph_vocab)
        document.tag_ner(ner_tagger)
        document.parse_syntax(syntax_parser)

        res = []
        for span in document.spans:
            span.normalize(morph_vocab)
            if span.type == 'LOC':
                res.append(span.normal)

        self.city = None if ' '.join(res) == '' else ' '.join(res)
        # return self.city

    def get_weather(self):
        lat = self.lat
        lon = self.lon
        city = self.city
        period = self.period
        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=hourly,current,minutely,alerts&units=metric&appid={api_token}&lang=ru'
        r = requests.get(url)
        weather_list = r.json()['daily']
        weather_list = [day_dict_proc(day) for day in weather_list]

        print('len weather_list (0)', len(weather_list))

        for i, day_weather in enumerate(weather_list):
            if self.date == day_weather[0]:
                break

        weather_list = weather_list[i: i + period]

        print('len weather_list (1)', len(weather_list))

        if len(weather_list) == 1:
            weather = weather_list[0]
            res = [f'Погода {weather[0]:%d-%m-%Y} ({city}):']
            text = f'Температура воздуха {weather[1]}°C (ощущается как {weather[2]}°C), {weather[4]}. Влажность воздуха {weather[3]}.'
            res.append(text)
            return '\n'.join(res)
        else:
            for i in range(len(weather_list)):
                weather_list[i][0] = f"{weather_list[i][0]:%d-%m-%Y}"
                vals = weather_list[i]
                weather_list[i] = f'{vals[0]}: {vals[1]}°C, {vals[4]}'

        res = [f'Погода ({city}):']
        res.extend(weather_list)
        return '\n'.join(res)

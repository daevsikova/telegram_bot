import telebot
import pymorphy2
import nltk
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deeppavlov import configs, build_model
from parse.horoscope_parser import HoroscopeParser
from parse.user import User

bot = telebot.TeleBot(config.BOT_TOKEN)

tokenizer_tox = AutoTokenizer.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
toxic_model = AutoModelForSequenceClassification.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

morph = pymorphy2.MorphAnalyzer()
tokenizer = nltk.tokenize.TreebankWordTokenizer()

user_dict = {}


@bot.message_handler(commands=['start'])  # Функция отвечает на команду 'start'
def start_message(message):
    bot.send_message(message.chat.id,
                     f"Привет!\n"
                     f"• Я знаю <b>погоду</b> в любом городе мира ☀\n"
                     f"• Подскажу твой <b>гороскоп</b> 🔮\n"
                     f"• А также посоветую <b>рецепт</b> из желаемых ингредиентов️🍝\n\n"
                     f"Чтобы узнать полный список команд, напиши /help \n"
                     f"Чтобы закончить диалог, напиши /exit\n",
                     parse_mode='HTML')


@bot.message_handler(commands=['help'])  # Функция отвечает на команду 'help'
def help_message(message):
    bot.send_message(message.chat.id,
                     f"<b>Я знаю следующие команды</b>:\n\n"
                     f"/help - <i>Повторить это сообщение</i>\n\n"
                     f"/exit - <i>Выход</i>\n\n"
                     f"Если хочешь узнать погоду, гороскоп или интересный рецепт -- напиши мне об этом, я тебя пойму!",
                     parse_mode='HTML')


@bot.message_handler(commands=['exit'])  # Функция отвечает на команду 'exit'
def end_message(message):
    bot.send_message(message.chat.id,
                     f"Рад был помочь! До встречи!\n")


@bot.message_handler(content_types=['text'])  # Функция обрабатывает текстовые сообщения
def get_text(message):
    user = User()
    user_dict[message.chat.id] = user_dict.get(message.chat.id, user)

    request_words = tokenize_text(message.text)
    if 'овен' not in request_words:
        # check for toxic message
        tokens_pt = tokenizer_tox(message.text, return_tensors="pt")
        with torch.no_grad():
            pred = torch.nn.functional.softmax(toxic_model(**tokens_pt)[0], dim=1).squeeze()
            request_is_toxic = pred[1] > 0.8

        if request_is_toxic:
            bot.send_message(message.chat.id,
                             text='Очень грубо 🗿😤 Я к такому не привыкла!\n\nЧтобы вызвать список команд, введите /help')
            return

    # classify request
    cnt_keywords_horo = sum([1 if word in HoroscopeParser.keywords else 0 for word in request_words])

    if cnt_keywords_horo > 0:
        # try to find date
        horo_parse = HoroscopeParser()
        horo_date = horo_parse.process_date(message.text, ner_model)
        user_dict[message.chat.id].horo_date = horo_date

        # try to find horo sign
        horo_sign = horo_parse.process_sign(message.text)
        if horo_sign is None:
            msg = bot.reply_to(message, 'Назови знак зодиака 🔮')
            bot.register_next_step_handler(msg, process_sign_step)
        else:
            user_dict[message.chat.id].horo_sign = horo_sign
            generate_horo(message)

    else:
        bot.send_message(message.chat.id,
                         text='Я тебя не понимаю(\n\nЧтобы вызвать список команд, введите /help')


def process_sign_step(message):
    try:
        horo_parse = HoroscopeParser()
        chat_id = message.chat.id
        horo_sign = horo_parse.process_sign(message.text)
        if horo_sign is None:
            msg = bot.reply_to(message, 'Попробуй еще! Назови знак зодиака 🔮')
            bot.register_next_step_handler(msg, process_sign_step)
            return
        user_dict[chat_id].horo_sign = horo_sign
        generate_horo(message)
    except Exception as e:
        bot.reply_to(message, 'Что-то пошло не так...')


def generate_horo(message):
    try:
        horo_parse = HoroscopeParser()
        chat_id = message.chat.id
        # get horoscope
        horo_date, horo_sign = user_dict[chat_id].horo_date, user_dict[chat_id].horo_sign
        bot.send_message(chat_id,
                         text='Предсказание почти готово... 🧙‍♀️\n')
        final_horo = horo_parse.get_horo(horo_date, horo_sign)
        # send horoscope to user
        bot.send_message(chat_id,
                         text=final_horo)

        # clear for opportunity to get new horo
        user_dict[chat_id].horo_date, user_dict[chat_id].horo_sign = None, None
        bot.send_message(chat_id,
                         text='Я могу еще чем-то помочь?\nЕсли нет, то попрощайся со мной или напиши /exit')
    except Exception as e:
        bot.reply_to(message, 'Что-то пошло не так...')


def tokenize_text(text):
    words = tokenizer.tokenize(text)
    result = []

    for word in words:
        p = morph.parse(word)[0]
        result.append(p.normal_form)
    return result

bot.polling(none_stop=True, interval=0)

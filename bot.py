import telebot
import pymorphy2
import nltk
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deeppavlov import configs, build_model

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
                     f"/weather - <i>Узнать погоду</i>\n\n"
                     f"/horo - <i>Узнать свой гороскоп</i>\n\n"
                     f"/cook - <i>Получить рецепт по желаемым ингредиентам</i>\n\n"
                     f"/exit - <i>Выход</i>\n",
                     parse_mode='HTML')


@bot.message_handler(commands=['exit'])  # Функция отвечает на команду 'exit'
def end_message(message):
    bot.send_message(message.chat.id,
                     f"Рад был помочь! До встречи!\n")


@bot.message_handler(content_types=['text'])  # Функция обрабатывает текстовые сообщения
def get_text(message):
    tokens_pt = tokenizer_tox(message.text, return_tensors="pt")
    with torch.no_grad():
        request_is_toxic = torch.argmax(toxic_model(**tokens_pt)[0]).item()

    if request_is_toxic:
        bot.send_message(message.chat.id,
                         text='Очень грубо 🗿😤 Я к такому не привыкла!\n\nЧтобы вызвать список команд, введите /help')
        return

    request_words = tokenize_text(message.text)
    # здесь будет функциональность

    bot.send_message(message.chat.id,
                     text='Скоро всё будет готово 😏\n')
    bot.send_message(message.chat.id,
                    text='Я могу еще чем-то помочь?\nЕсли нет, то попрощайся со мной или напиши /exit')


def tokenize_text(text):
    words = tokenizer.tokenize(text)
    result = []

    for word in words:
        p = morph.parse(word)[0]
        result.append(p.normal_form)
    return result


bot.polling(none_stop=True, interval=0)

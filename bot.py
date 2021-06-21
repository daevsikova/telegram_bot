import telebot
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from deeppavlov import configs, build_model
from parse.recipe_parser import RecipeParser
from parse.horoscope_parser import HoroscopeParser
from parse.weather_parser import WeatherParser
from parse.user import User
from utils import log_info, check_coordinates

bot = telebot.TeleBot(config.BOT_TOKEN)

tokenizer_tox = AutoTokenizer.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
toxic_model = AutoModelForSequenceClassification.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

user_dict = {}

recipe_parser = RecipeParser()
horoscope_parser = HoroscopeParser()
weather_parser = WeatherParser()

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


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
    user_dict[message.chat.id].needs_greet = True
    bot.send_message(message.chat.id,
                     f"Рада была помочь! До встречи!\n")


def is_toxic(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    for token in message.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma.lower() in horoscope_parser.horo_signs:
            return 0
    tokens_pt = tokenizer_tox(message.text, return_tensors="pt")
    with torch.no_grad():
        pred = torch.nn.functional.softmax(toxic_model(**tokens_pt)[0], dim=1).squeeze()
        request_is_toxic = pred[1] > 0.8
        return request_is_toxic


def is_appology(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    cnt = 0
    for token in message.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma.lower() in ['извинить', 'извинение', 'простить', 'прощение', 'извини']:
            cnt += 1
    return cnt


def is_bye(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    cnt = 0
    for token in message.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma.lower() in ['пока', 'свидание']:
            cnt += 1
    return cnt


@bot.message_handler(content_types=['text'])  # Функция обрабатывает текстовые сообщения
def get_text(message):
    # try:
        chat_id = message.chat.id
        tox = is_toxic(message)
        if tox:
            if chat_id in user_dict:
                user_dict[chat_id].toxic = tox
            else:
                user_dict[chat_id] = User(tox)
            bot.reply_to(message,
                        f"Очень грубо 🗿😤 Я к такому не привыкла!\n\n"
                        f"Не буду вам помогать, пока не извинитесь..."
                        )
            # bot.register_next_step_handler(msg, get_text)
            return

        if chat_id in user_dict and user_dict[chat_id].toxic:
            if is_appology(message):
                bot.reply_to(message, "Ваши извинения приняты! Теперь можете просить меня о чем угодно!")
                user_dict[chat_id].toxic = 0
            else:
                bot.reply_to(message, "Я по-прежнему жду ваших извинений...")
            # bot.register_next_step_handler(msg, get_text)
            return
        if chat_id not in user_dict:
            user_dict[chat_id] = User(tox)
        if is_bye(message):
            user_dict[chat_id].needs_greet = True
            bot.send_message(chat_id, "Рада была помочь! До встречи!")
            return
        if user_dict[chat_id].needs_greet:
            user_dict[chat_id].needs_greet = False
            bot.send_message(chat_id, "Приветики-пистолетики!")

        # query classification
        weather_cnt, horoscope_cnt, recipe_cnt = 0, 0, 0
        doc = Doc(message.text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            if token.lemma in weather_parser.keywords:
                weather_cnt += 1
            elif token.lemma in horoscope_parser.keywords:
                horoscope_cnt += 1
            elif token.lemma in recipe_parser.keywords:
                recipe_cnt += 1

        if (weather_cnt > 0) + (horoscope_cnt > 0) + (recipe_cnt > 0) > 1:
            bot.reply_to(message, 'Я умею делать только что-то одно за раз!\n'
                                  'Попросите меня еще раз, но только о чем-то одном')
            return
        if weather_cnt + horoscope_cnt + recipe_cnt == 0:
            bot.reply_to(message, 'Я не поняла, чего вы от меня хотите((((\n'
                                  'Спросите, пожалуйста, еще раз как-нибудь по-другому')
            return
        if weather_cnt > 0:
            # bot.reply_to(message, 'Держите ваш прогноз:')
            process_weather_step(message)
            return
        if horoscope_cnt > 0:
            bot.reply_to(message, 'Вот о чем мне рассказали звезды:')
            process_horoscope_step(message)
            return
        bot.reply_to(message, 'Могу предложить такой вариантик:')
        process_recipe_step(message)
    # except Exception as e:
    #     bot.reply_to(message, 'Что-то пошло не так...')


def process_horoscope_step(message):
    horo_date = horoscope_parser.process_date(message.text, ner_model)
    user_dict[message.chat.id].horo_date = horo_date

    # try to find horo sign
    horo_sign = horoscope_parser.process_sign(message.text)
    if horo_sign is None:
        msg = bot.reply_to(message, 'Назови знак зодиака 🔮')
        bot.register_next_step_handler(msg, process_sign_step)
    else:
        user_dict[message.chat.id].horo_sign = horo_sign
        generate_horo(message)


def process_sign_step(message):
    chat_id = message.chat.id
    horo_sign = horoscope_parser.process_sign(message.text)
    if horo_sign is None:
        msg = bot.reply_to(message, 'Попробуй еще! Назови знак зодиака 🔮')
        bot.register_next_step_handler(msg, process_sign_step)
        return
    user_dict[chat_id].horo_sign = horo_sign
    generate_horo(message)


def generate_horo(message):
    chat_id = message.chat.id
    # get horoscope
    horo_date, horo_sign = user_dict[chat_id].horo_date, user_dict[chat_id].horo_sign
    bot.send_message(chat_id,
                     text='Предсказание почти готово... 🧙‍♀️\n')
    final_horo = horoscope_parser.get_horo(horo_date, horo_sign)
    # send horoscope to user
    bot.send_message(chat_id,
                     text=final_horo)

    # clear for opportunity to get new horo
    user_dict[chat_id].horo_date, user_dict[chat_id].horo_sign = None, None
    bot.send_message(chat_id,
                     text='Я могу еще чем-то помочь?\nЕсли нет, то попрощайся со мной или напиши /exit')


def process_recipe_step(message):
    chat_id = message.chat.id
    recipe_ingredients = recipe_parser._extract_ingredients(Doc(message.text))
    if len(recipe_ingredients) == 0:
        bot.reply_to(message, 'Я не знаю таких ингридиентов(((\nСпросите меня рецепт из чего-нибудь другого.')
        # bot.register_next_step_handler(msg, get_text)
    else:
        recipe = recipe_parser.process(recipe_ingredients)
        if recipe is None:
            bot.reply_to(message, 'Мне не удалось найти ни одного рецепта из указаных вами ингридиентов((('
                                  '\nСпросите меня рецепт из чего-нибудь другого.')
            # bot.register_next_step_handler(msg, get_text)
        else:
            out_msg = format_recipe(recipe)
            bot.reply_to(message, out_msg, parse_mode='HTML')
            bot.send_message(chat_id,
                             text='Я могу еще чем-то помочь?\nЕсли нет, то попрощайся со мной или напиши /exit')
            # bot.register_next_step_handler(msg, get_text)


def format_recipe(recipe):
    out = f"<b>Все необходимые ингридиенты:</b>\n\n"
    for ingredient in recipe['ingredients']:
        out += f"{ingredient}\n"
    out += f"\n\n<b>Шаги приготовления:</b>\n\n"
    for i, step in enumerate(recipe['steps']):
        out += f"Шаг {i+1}:\n{step}\n\n"
    return out

#########################################################
def process_weather_step(message): # главная функция, которая управляет всеми сценариями
    user = user_dict[message.chat.id]

    weather_parser.get_date(message.text, ner_model)
    weather_parser.get_city(message.text)

    if weather_parser.period is None:
        weather_parser.get_period(message.text, ner_model)

    city = weather_parser.city
    period = weather_parser.period
    print(log_info(weather_parser))

    scenario1 = period and city
    scenario2 = period and (city is None)
    scenario3 = (period is None) and city
    scenario4 = (period is None) and (city is None)

    if scenario2:
        try:
            msg = bot.reply_to(message, 'Я могу сказать погоду но ты не указал в каком городе. Укажешь?')
            bot.register_next_step_handler(msg, process_city_step)
        except Exception as e:
            bot.reply_to(message, 'Так, у меня возникли какие-то проблемы. Давай попробуем всё заново.')

    elif scenario3:
        try:
            msg = bot.reply_to(message, 'Я могу сказать погоду но ты не указал период (сколько дней). Укажешь?')
            bot.register_next_step_handler(msg, process_period_step)
        except Exception as e:
            bot.reply_to(message, 'Так, у меня возникли какие-то проблемы. Давай попробуем всё заново.')

    elif scenario4:
        try:
            msg = bot.reply_to(message,
                               'Я могу сказать погоду но ты не указал период (сколько дней) и город. Для начала можешь указать город?')
            bot.register_next_step_handler(msg, process_city_step)
        except Exception as e:
            bot.reply_to(message, 'Так, у меня возникли какие-то проблемы. Давай попробуем всё заново.')

    else:  # scenario1
        if weather_parser.city and weather_parser.period:
            # меняем город, если проблемы с координатами
            res = check_coordinates(weather_parser.city)
            if len(res) == 2:
                weather_parser.lat, weather_parser.lon = res
                text = weather_parser.get_weather()
                print(log_info(weather_parser))
                # обнулим всё в парсере
                weather_parser.city = None
                weather_parser.lat = None
                weather_parser.lon = None
                weather_parser.period = None
                weather_parser.date = None
                bot.send_message(message.from_user.id, text)
            else:
                weather_parser.city = None
                msg = bot.reply_to(message, res)
                bot.register_next_step_handler(msg, process_city_step)


def process_city_step(message):
    chat_id = message.chat.id
    user = user_dict[chat_id]

    weather_parser.get_city(message.text)
    print(log_info(weather_parser))

    if weather_parser.city is None:
        msg = bot.reply_to(message,
                           'Ты уверен, что такой город существует? В моих базах его нет, попробуй ввести город ещё раз')
        bot.register_next_step_handler(msg, process_city_step)
        return

    if weather_parser.city and weather_parser.period:
        bot.reply_to(message, 'Отлично! Такой город я знаю')
        # меняем город, если проблемы с координатами
        res = check_coordinates(weather_parser.city)
        if len(res) == 2:
            weather_parser.lat, weather_parser.lon = res
            text = weather_parser.get_weather()
            print(log_info(weather_parser))
            # обнулим всё в парсере
            weather_parser.city = None
            weather_parser.lat = None
            weather_parser.lon = None
            weather_parser.period = None
            weather_parser.date = None
            bot.send_message(message.from_user.id, text)
        else:
            weather_parser.city = None
            msg = bot.reply_to(message, res)
            bot.register_next_step_handler(msg, process_city_step)
    elif weather_parser.city:
        msg = bot.reply_to(message, 'Отлично! Такой город я знаю. Можешь теперь ещё уточнить период?')
        bot.register_next_step_handler(msg, process_period_step)


def process_period_step(message):
    chat_id = message.chat.id
    user = user_dict[chat_id]

    weather_parser.get_period(message.text, ner_model)
    print(log_info(weather_parser))

    if weather_parser.period is None:
        msg = bot.reply_to(message, 'Ты уверен, что это корректный период? Что-то я его не понимаю. Попробуй ещё раз')
        bot.register_next_step_handler(msg, process_period_step)
        return

    bot.reply_to(message, 'Отлично! Понимаю о чём ты')
    if weather_parser.city and weather_parser.period:
        # меняем город, если проблемы с координатами
        res = check_coordinates(weather_parser.city)
        if len(res) == 2:
            weather_parser.lat, weather_parser.lon = res
            text = weather_parser.get_weather()
            print(log_info(weather_parser))

            # обнулим всё в парсере
            weather_parser.city = None
            weather_parser.lat = None
            weather_parser.lon = None
            weather_parser.period = None
            weather_parser.date = None
            bot.send_message(message.from_user.id, text)
        else:
            weather_parser.city = None
            msg = bot.reply_to(message, res)
            bot.register_next_step_handler(msg, process_city_step)
#########################################################


bot.polling(none_stop=True, interval=0)

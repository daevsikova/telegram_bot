import requests
import datetime

def day_dict_proc(day_dict):
    temp = int(day_dict['temp']['day'])
    fl_temp = int(day_dict['feels_like']['day'])
    hum = day_dict['humidity']
    desc = day_dict['weather'][0]['description']
    data = datetime.datetime.fromtimestamp(int(day_dict['dt'])).date()
    return [data, temp, fl_temp, hum, desc]

def log_info(parser):
    text = f'LOG: date: {parser.date}, city: {parser.city}, lat: {parser.lat}, lon: {parser.lon}, period: {parser.period}'
    return text

def check_coordinates(city):
    try:
        url = f'https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={api_token}&lang=ru'
        r = requests.get(url).json()[0]
        lat = r['lat']
        lon = r['lon']
        return lat, lon
    except:
        return 'Случшай, я не смог получить координаты для этого города. Хотя он мне известен, возможно это неполадки на сервере. Можешь ввести название другого города'
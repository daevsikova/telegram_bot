class User:
    def __init__(self, toxic):
        self.toxic = toxic
        self.weather_data = None
        self.horo_sign = None
        self.horo_date = None
        self.needs_greet = True  # нужно ли здороваться с пользователем

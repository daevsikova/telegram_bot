from parse.command_parser import CommandParser
from natasha import Doc

# hello

class WeatherParser(CommandParser):

    def __init__(self):
        self.keywords = ['погода', 'температура']

    def process(self, message: Doc):
        pass
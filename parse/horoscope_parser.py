from parse.command_parser import CommandParser
from natasha import Doc


class HoroscopeParser(CommandParser):
    def __init__(self):
        self.keywords = ['гороскоп']

    def process(self, message: Doc):
        pass

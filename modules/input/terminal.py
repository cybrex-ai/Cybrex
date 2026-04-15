from interfaces import InputInterface

class Module(InputInterface):
    def get_input(self):
        return input("\n> ")
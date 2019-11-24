class Plotter:
    def __init__(self):
        self.interfaces = []
        self.abstract_plots = {}

    def add(self, interface, *args, **kwargs):
        self.abstract_plots.append(interface.read(*args, **kwargs))

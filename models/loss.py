import numpy as np


class lossCalculator:

    def __init__(self, mode):
        print 'Initializing a loss calculator ...'
        self.mode = mode

    def update_mode(self, mode):
        self.mode = mode

    def calculate(self, predictions, labels):
        if self.mode.lower() == 'mse':
            return np.mean(np.square(predictions - labels))


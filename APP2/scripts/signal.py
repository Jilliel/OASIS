import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import numpy as np

class Signal:
    def __init__(self, data, Fe):
        self.data = data
        self.Fe = Fe

    def show_fft(self, down=None, up=None):
        data = self.data[down:up]
        fourrier = fft(data)
        freq = fftfreq(len(data), d=1/self.Fe)
        plt.plot(freq, np.abs(fourrier))
        plt.xlabel("Fréquence réduite")
        plt.ylabel("Module de la fft")
        plt.show()

    
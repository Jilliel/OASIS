import numpy as np

class Notes:
    def __init__(self):
        self.notes = np.array([32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74])
        self.names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

    def up(self):
        self.notes = 2 * self.notes

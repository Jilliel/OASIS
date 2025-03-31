import numpy as np

class Notes:
    MAXFREQ = 4000
    def __init__(self):
        self.notes = np.array([32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74])
        self.names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

    def isValid(self):
        return self.notes[-1] < Notes.MAXFREQ
    
    def items(self):
        return zip(self.names, self.notes)
    
    def reset(self):
        self.notes = np.array([32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74])

    def up(self):
        self.notes = 2 * self.notes

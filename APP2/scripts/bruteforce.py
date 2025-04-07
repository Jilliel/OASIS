import matplotlib.pyplot as plt
from scipy.signal import stft
import numpy as np

VERBOSE = True
TRESHOLD_NOTE = 0.9
TRESHOLD_INSTRUMENT = 0.5
TRESHOLD_BOTH = 0.1
EPSILON = 1e-9
REP = "audio/"
EXT = ".wav"
SAVE = "results/"

def plotSpecter(f, t, Z):
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, np.abs(Z), shading='gouraud')
    plt.ylim(0, 2000)
    plt.ylabel('Fréquence')
    plt.xlabel('Temps')
    plt.title('Spectrogramme de la tftc')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig("result.png")

class Notes:
    MINFREQ = 32.7
    MAXFREQ = 3951.36
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

def analyse(x, Fe):
    reference = Notes()
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    N = 2048
    zpad = 10
    frequences, time, Zxx = stft(x, fs=Fe, nperseg=N, nfft=zpad*N)

    n = len(reference.notes)
    m = time.shape[0]
    
    
    history = np.zeros((m, n), dtype=np.uint)
    instruments = np.zeros((m, 2)) #0: Guitare, 1: Percussions
    former = [] #Energie max à chaque instant

    if VERBOSE:
        print("Nombre de steps: ", m)
    all_notes = {}
    for t in range(m):
        all_notes[t] = []
        fft = np.abs(Zxx[:, t]) # Fft à l'instant time[t]
        if VERBOSE:
            print("Instant: ", time[t])

        # On détermine la note
        reference.reset()
        energy = {name: 0 for name in reference.names}
        while reference.isValid():
            for name, note in reference.items():
                df = np.abs(frequences-note)
                energy[name] += fft[np.argmin(df)]
            reference.up()

        e_max = max(energy.values())
        former.append(e_max) # On enregistre l'energie

        for index, name in enumerate(reference.names):
            if energy[name]/e_max > TRESHOLD_NOTE:
                history[t][index] = 1
                all_notes[t].append(name)

    # On annule les valeurs dont l'energie est trop faible
    threshold = np.mean(former) / 2
    for t in range(m):
        if former[t] <= threshold:
            instruments[t][0] = 0
            instruments[t][1] = 0
            for i in range(len(reference.names)):
                history[t][i] = 0
            all_notes[t] = []


    delta_t = 5
    for t, L in all_notes.items():
        notes_near = {}
        for tb in range(t-delta_t, t+delta_t+1):
            instant = all_notes.get(tb, None)
            if instant is not None:
                for n in instant:
                    if n in notes_near:
                        notes_near[n] += 1
                    else:
                        notes_near[n] = 1

        for n in L :
            if n in notes_near and notes_near[n] >4 :
                instruments[t][0] = 1
            else:
                instruments[t][1] = 1

        if VERBOSE :
            print(f"Instant {np.round(time[t], 2)}, guitare : {bool(instruments[t][0])}, percussion : {bool(instruments[t][0])}.")

    return history, instruments
    #plt.plot(time, ratios, color="black")
    #plt.plot((time[0], time[-1]), (TRESHOLD_INSTRUMENT-TRESHOLD_BOTH, TRESHOLD_INSTRUMENT-TRESHOLD_BOTH), color="red")
    #plt.plot((time[0], time[-1]), (TRESHOLD_INSTRUMENT+TRESHOLD_BOTH, TRESHOLD_INSTRUMENT+TRESHOLD_BOTH), color="red")
    #plt.xlabel("Time")
    #plt.ylabel("Ratio")
    #plt.ylim(0, 1)
    #plt.savefig(SAVE+"ratios.png")
    #plt.clf()
    #lt.cla()
    #plt.imshow(history)
    #plt.gca().invert_yaxis()
    #plt.savefig(SAVE+"hist.png")
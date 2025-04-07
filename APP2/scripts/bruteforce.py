import matplotlib.pyplot as plt
from scripts.utils import Notes
from scipy.signal import stft
import soundfile as sf
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


def analyseNoteRaphael(file):
    reference = Notes()
    x, Fe = sf.read(REP+file+EXT)
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
    ratios = []
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
                if False:
                    print(f"Note: {name} ; Energy: {energy[name]}")

        # On détermine l'instrument
        mask = (Notes.MINFREQ <= frequences) & (frequences < Notes.MAXFREQ)
        sample = fft[mask]
        arithmetic_mean = np.mean(sample)
        geometric_mean = np.exp(np.mean(np.log(sample + EPSILON)))
        ratio = geometric_mean / (arithmetic_mean + EPSILON)
        ratios.append(ratio)

        if ratio > TRESHOLD_INSTRUMENT + TRESHOLD_BOTH:
            instruments[t][1] = 1
        elif ratio < TRESHOLD_INSTRUMENT - TRESHOLD_BOTH:
            instruments[t][0] = 1
        else: 
            instruments[t][0] = 1
            instruments[t][1] = 1


    # On annule les valeurs dont l'energie est trop faible
    threshold = np.mean(former) / 10
    for t in range(m):
        if former[t] <= threshold:
            ratios[t] = 0
            instruments[t][0] = 0
            instruments[t][1] = 0
            for i in range(len(reference.names)):
                history[t][i] = 0
            all_notes[t] = []
    guitare = {}
    percussions = {}
    delta_t = 5
    for t, L in all_notes.items():
        guitare[t] = []
        percussions[t] = []
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
                guitare[t].append(n)
            else:
                percussions[t].append(n)
        if VERBOSE :
            print(f"Instant {np.round(time[t], 2)}, guitare : {guitare[t]}, percussion : {percussions[t]}")

    plt.plot(time, ratios, color="black")
    plt.plot((time[0], time[-1]), (TRESHOLD_INSTRUMENT-TRESHOLD_BOTH, TRESHOLD_INSTRUMENT-TRESHOLD_BOTH), color="red")
    plt.plot((time[0], time[-1]), (TRESHOLD_INSTRUMENT+TRESHOLD_BOTH, TRESHOLD_INSTRUMENT+TRESHOLD_BOTH), color="red")
    plt.xlabel("Time")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.savefig(SAVE+file+"-ratios.png")
    plt.clf()
    plt.cla()
    #plt.imshow(history)
    #plt.gca().invert_yaxis()
    #plt.savefig(FILE+"-hist.png")
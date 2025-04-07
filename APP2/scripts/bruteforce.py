import matplotlib.pyplot as plt
from scripts.utils import Notes
from scipy.signal import stft
import soundfile as sf
import numpy as np

TRESHOLD = 0.9
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
    ratios = []
    print("Nombre de steps: ", m)

    for t in range(m):
        
        fft = np.abs(Zxx[:, t])
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
        for index, name in enumerate(reference.names):
            if energy[name]/e_max > TRESHOLD:
                history[t][index] = 1
                print(f"Note: {name} ; Energy: {energy[name]}")
    
        # On détermine l'instrument
        mask = (100 <= frequences) & (frequences < 5000)
        sample = fft[mask]
        arithmetic_mean = np.mean(sample)
        geometric_mean = np.exp(np.mean(np.log(sample + EPSILON)))
        ratios.append(geometric_mean / (arithmetic_mean + EPSILON))

    plt.plot(time, ratios)
    plt.xlabel("Time")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.savefig(SAVE+file+"-ratios.png")
    plt.clf()
    plt.cla()
    #plt.imshow(history)
    #plt.gca().invert_yaxis()
    #plt.savefig(FILE+"-hist.png")
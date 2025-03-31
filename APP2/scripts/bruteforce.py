import matplotlib.pyplot as plt
from scripts.utils import Notes
from scipy.signal import stft
import soundfile as sf
import numpy as np

def plot(f, t, Z):
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, np.abs(Z), shading='gouraud')
    plt.ylim(0, 2000)
    plt.ylabel('Fréquence')
    plt.xlabel('Temps')
    plt.title('Spectrogramme de la tftc')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig("result.png")


def analyse():
    reference = Notes()
    x, Fe = sf.read("audio/gamme_demiTon_guitare.wav")
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    frequences, time, Zxx = stft(x, fs=Fe, nperseg=1024)

    for t in time:
        fft = np.abs(Zxx[t])
        reference.reset()
        energy = {name: 0 for name in reference.names}
        while reference.isValid():
            for name, note in reference.items():
                df = np.abs(frequences-note)
                i = np.argmax(df)
                energy[name] += fft[i]
            reference.up()
        input(energy)


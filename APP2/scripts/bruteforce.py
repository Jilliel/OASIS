import matplotlib.pyplot as plt
from scripts.utils import Notes
from scipy.signal import stft
import soundfile as sf
import numpy as np

TRESHOLD = 0.9

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
    x, Fe = sf.read("audio/love_me_SI101_drums.wav")
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    N = 2048
    zpad = 10
    frequences, time, Zxx = stft(x, fs=Fe, nperseg=N, nfft=zpad*N)
    steps = time.shape[0]

    print("Nombre de steps: ", steps)

    for t in range(steps):
        
        fft = np.abs(Zxx[:, t])
        reference.reset()
        energy = {name: 0 for name in reference.names}
        while reference.isValid():
            for name, note in reference.items():
                df = np.abs(frequences-note)
                i = np.argmin(df)
                energy[name] += fft[i]
            reference.up()
        print(f"Step: {t}", end = " / ")

        e_max = max(energy.values())
        notes = [name for name in energy if energy[name]/e_max > TRESHOLD]
        print(f"Notes: {notes}")


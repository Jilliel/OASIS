import numpy as np
from numpy.fft import fft
import soundfile as sf
import matplotlib.pyplot as plt

CHAR_LEN = 2000
CHAR_SEP = 500
CHAR_TOTAL = CHAR_SEP + CHAR_LEN

Nfft = 10000
CEIL = 0.3

def open(file):
    x, Fe = sf.read(file)
    x = x.reshape(x.shape[0], 1)
    return x, Fe

def f2char(f):
    if f > 0:
        return chr(f-436)
    else:
        return " "
    
def filtrage(x_fft, freq):
    x_fft[freq < 501] = 0
    x_fft[freq > 526] = 0

def decode_single(x, Fe) -> int:
    #hamming_window = np.hamming(CHAR_LEN)
    #hamming_window = hamming_window.reshape((CHAR_LEN, 1))
    #x_hamming = x * hamming_window
    #x_fft = fft(x_hamming, axis=0, n=Nfft*CHAR_LEN)
    x_fft = fft(x, axis=0, n=Nfft*CHAR_LEN)
    freq = np.linspace(0, 1, Nfft*CHAR_LEN) * Fe

    filtrage(x_fft, freq)

    i = np.argmax(np.abs(x_fft))
    return round(freq[i]), np.abs(x_fft[i])

def decode_multiple(x, Fe):
    n_f = round(x.shape[0] / CHAR_TOTAL)
    L_f = np.zeros((n_f, 1))
    L_mod = np.zeros((n_f, 1))
    for i in range(n_f):
        start = i * CHAR_TOTAL
        f, mod = decode_single(x[start:start+CHAR_LEN], Fe)
        L_f[i] = f
        L_mod[i] = mod

    L_mod = L_mod / np.mean(L_mod)
    L_f[L_mod < CEIL] = 0

    return [int(f[0]) for f in L_f]


def decode(file):
    x, Fe = open(file)
    print(f"Nombre d'échantillons: {x.shape[0]}")
    L_f = decode_multiple(x, Fe)
    print(L_f)
    return "".join([f2char(f) for f in L_f])


def test(file):
    x, Fe = open(file)
    width = x.shape[0]
    x = x.reshape(width, 1)
    hamming = np.hamming(x.shape[0]).reshape(width, 1)
    x_hamming = x * hamming
    nfft = 500
    freq = np.linspace(0, 1, nfft*width) * Fe
    x_fft = fft(x_hamming, nfft*width, axis=0)

    plt.plot(freq, np.abs(x_fft))
    plt.xlim(490, 530)
    plt.show()

if __name__ == "__main__":
    #print(decode("notebooks/mess_ssespace.wav"))
    #print(decode("notebooks/mess.wav")) 
    #print(decode("notebooks/mess_difficile.wav"))
    test("notebooks/mess_difficile.wav")
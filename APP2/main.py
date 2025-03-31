import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import soundfile as sf
import matplotlib.pyplot as plt
from scripts.bruteforce import analyse

def test1():
    # Dictionnaire des fréquences des notes (A4 = 440 Hz comme référence)
    notes = {
        'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
        'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
        'A#4': 466.16, 'B4': 493.88, 'C5': 523.25
    }

    def apply_filter(signal, Fe, f0, Q=10):
        """Applique un filtre passe-bande centré sur f0."""
        w0 = 2 * np.pi * f0 / Fe
        alpha = np.sin(w0) / (2 * Q)
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
        return sig.lfilter(b, a, signal)

    def detect_note(signal, Fe):
        """Applique les filtres et détecte la note dominante."""
        note_strength = {}
        for note, f0 in notes.items():
            filtered_signal = apply_filter(signal, Fe, f0)
            note_strength[note] = np.sum(filtered_signal ** 2)  # Énergie du signal filtré
        
        return max(note_strength, key=note_strength.get)

    def detect_instrument(signal, Fe):
        """Détecte si l'instrument est une guitare ou une percussion."""
        energy = np.sum(signal ** 2)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))))
        
        # Une guitare produit des sons plus continus avec moins de variations brusques
        # Une percussion a une énergie brève et des changements plus soudains
        if zero_crossings > 5000 and energy > 0.01:
            return "Percussion"
        else:
            return "Guitare"

    def analyse(musique, Fe):
        """Analyse un signal audio et détecte les notes jouées et l'instrument."""
        N = len(musique)
        frame_size = Fe // 10  # Analyse sur des fenêtres de 100 ms
        num_frames = N // frame_size
        
        notes_detected = []
        instruments_detected = []
        
        for i in range(num_frames):
            frame = musique[i * frame_size: (i + 1) * frame_size]
            note = detect_note(frame, Fe)
            instrument = detect_instrument(frame, Fe)
            notes_detected.append(note)
            instruments_detected.append(instrument)
        
        return np.array(notes_detected), np.array(instruments_detected)

    # Exemple d'utilisation:
    signal, Fe = sf.read("APP2/audio/gamme_demiTon_guitare.wav")
    Notes, Instruments = analyse(signal, Fe)
    print("Notes détectées:", Notes)
    print("Instruments détectés:", Instruments)

def test2():
    analyse()

if __name__ == "__main__":
    #test1()
    test2()
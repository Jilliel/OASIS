import numpy as np
from scipy.signal import find_peaks

def detect_instrument(fft_data, dt):
    """
    Détecte la présence simultanée de percussions et de guitares dans un signal
    à partir d'une série de FFT et de l'intervalle temporel dt (en secondes)
    entre chaque FFT.
    
    Pour chaque fenêtre FFT, la fonction calcule :
      - Le rapport de l'énergie contenue dans les pics harmoniques (trouvés avec find_peaks)
        par rapport à l'énergie totale (harmonic_ratio).
      - L'énergie totale de la fenêtre (pour construire l'enveloppe énergétique).
    
    La présence de guitare est caractérisée par :
      - Un harmonic_ratio moyen élevé (indiquant une prédominance des harmoniques).
      - Une faible variance du harmonic_ratio dans le temps (stabilité).
    
    La présence de percussions est caractérisée par :
      - Des pics transitoires dans l'enveloppe énergétique, sur une très courte durée (un ou deux dt).
    
    Paramètres:
      fft_data (array-like): Tableau 2D (n_frames x n_freq) contenant les FFT successives.
      dt (float): Intervalle de temps (en secondes) entre chaque FFT consécutive.
      
    Retourne:
      dict: {
             "guitare": bool,         # True si les critères harmoniques et leur stabilité suggèrent la présence d'une guitare.
             "percussion": bool,      # True si des événements transitoires sont détectés.
            }
    """
    # S'assurer que fft_data est au moins un tableau 2D
    fft_data = np.atleast_2d(fft_data)
    n_frames, n_freq = fft_data.shape
    
    # Calculer la matrice d'amplitude (module)
    amplitude = np.abs(fft_data)
    
    # Pour chaque FFT, calcul du rapport harmonique et de l'énergie totale
    harmonic_ratios = []
    energy_envelope = []  # énergie totale par fenêtre
    for i in range(n_frames):
        frame = amplitude[i, :]
        energy = np.sum(frame)
        energy_envelope.append(energy)
        
        # Recherche des pics dans le spectre de la fenêtre (seuil à 10% du pic maximum peut-etre a redefinir)
        seuil=0.1
        peaks, _ = find_peaks(frame, height=seuil*np.max(frame))
        peak_energy = np.sum(frame[peaks])
        ratio = peak_energy / energy if energy > 0 else 0
        harmonic_ratios.append(ratio)
        
    harmonic_ratios = np.array(harmonic_ratios)
    energy_envelope = np.array(energy_envelope)
    
    # Analyse de la composante harmonique (guitare)
    avg_harmonic_ratio = np.mean(harmonic_ratios)
    temporal_variance = np.var(harmonic_ratios)
    
    # Critères indicatifs pour la guitare (peut-etre a ajuster)
    guitar_harmonic_threshold = 0.5   # au moins 50% d'énergie dans les pics harmoniques
    guitar_variance_threshold = 0.01    # stabilité importante dans le temps
    guitar_detected = (avg_harmonic_ratio > guitar_harmonic_threshold and 
                       temporal_variance < guitar_variance_threshold)
    
    # Analyse temporelle de l'enveloppe énergétique pour détecter des percussions transitoires
    # On cherche des pics dans l'enveloppe énergétique qui durent très peu de temps.
    # Ici, on détecte les pics globaux sur l'enveloppe :
    energy_peaks, peak_props = find_peaks(energy_envelope, height=0.1*np.max(energy_envelope))
    
    percussion_events = 0
    for idx in energy_peaks:
        # On considère un pic transitoire si les fenêtres avant et après ne sont pas élevées.
        # On vérifie si, dans une fenêtre de ±1 (i.e. environ 2*dt), le pic est isolé.
        if idx == 0 or idx == n_frames-1:
            duration = dt
        else:
            # Comparaison avec les voisins : si les voisins sont inférieurs à 50% du pic, c'est transitoire.
            if energy_envelope[idx-1] < 0.5 * energy_envelope[idx] and energy_envelope[idx+1] < 0.5 * energy_envelope[idx]:
                duration = dt  # l'événement semble tenir sur une fenêtre unique
            else:
                duration = 2 * dt  # potentiellement moins transitoire
        if duration <= dt:
            percussion_events += 1
    
    percussion_ratio = percussion_events / n_frames
    # Critère indicatif : si plus de 10% des fenêtres présentent un pic transitoire, on détecte des percussions.
    percussion_detected = percussion_ratio > 0.1
    
    result = {
        "guitare": guitar_detected,
        "percussion": percussion_detected,
    }
    
    return result

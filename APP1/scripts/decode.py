import numpy as np
import matplotlib.pyplot as plt

class Signal():
    def __init__(self, signal : np.ndarray, threshold = 0.3,  Fe = 8000,  symbol_size = 2000, void_size = 500):
        self.complete_signal = signal.reshape(len(signal), 1)
        self.Fe = Fe
        self.symbol_size = symbol_size
        self.void_size = void_size

        self.len = signal.shape[0]

        self.signal_char_list = []
        self.signal_void_list = []
        self.threshold = threshold

    def seperate(self):
        i = 0
        symbol = True
        while i < self.len :
            if symbol : 
                self.signal_char_list.append(self.complete_signal[i:i+self.symbol_size])
                i+= self.symbol_size
            else:
                self.signal_void_list.append(self.complete_signal[i:i+self.void_size])
                i+= self.void_size
            symbol = not symbol
    
    def decode(self, show=False):
        self.seperate()
        message = ''
        characters_list = []
        modules_list = []
        weighted_mean = 0
        total_weight = 0
        for encoded_char in self.signal_char_list :
            decoded_char, mod, j = Char(encoded_char, self.Fe).decode(show)
            characters_list.append(decoded_char)
            modules_list.append(mod)
            if decoded_char != " ":
                weighted_mean += mod /(j+1)
                total_weight += (1/(j+1))
        weighted_mean = weighted_mean / total_weight
        #print(f'Mean is {weighted_mean}')
        for i in range(len(characters_list)):
            if characters_list[i] != " ":
                r = modules_list[i]/weighted_mean
                if r > 1+self.threshold or r < 1-self.threshold :
                    #print(f'Adding a space as {characters_list[i]}\'s mod was too far from the mean with a ratio of {r}')
                    characters_list[i] = " "
                #else:
                    #print(f'Adding a {characters_list[i]} as the ratio was good enough : {r}')
            #else : 
                #(f'Adding a space')
            message = message + characters_list[i]
        return message

class Char():
    def __init__(self, signal : np.ndarray, Fe = 8000):
        self.complete_signal = signal.reshape(len(signal), 1)
        
        self.Fe = Fe
        self.len = signal.shape[0]
      
        self.freq_to_char = {0: ' ', 501: 'A', 502: 'B', 503: 'C', 504: 'D', 505: 'E', 506: 'F', 507: 'G', 508: 'H', 509: 'I', 510: 'J', 511: 'K', 512: 'L', 513: 'M', 514: 'N', 515: 'O', 516: 'P', 517: 'Q', 518: 'R', 519: 'S', 520: 'T', 521: 'U', 522: 'V', 523: 'W', 524: 'X', 525: 'Y', 526: 'Z'}
        self.zero_padd = 128
        self.max_removals = 15

        self.complete_signal_fft = np.fft.fft(self.complete_signal, axis= 0, n= self.zero_padd * self.len, norm='ortho')

    def get_symbol(self, freq):
        return self.freq_to_char[freq]
    
    def show_signal(self):
        plt.plot(range(self.len), self.complete_signal)
        plt.show()

    def show_fft_signal(self):
        f = np.fft.fftfreq(self.zero_padd * self.len, d=1/self.Fe)
        #signal_fft = np.fft.fft(self.complete_signal, axis= 0, n= self.zero_padd * self.len, norm='ortho')
        mask = (f >= 501) & (f <= 526)
        plt.plot(f[mask], np.abs(self.complete_signal_fft[mask]) )
        plt.show()

    def get_biggest_frequency(self):
        """
        On calcule la TF, on regarde son module et on prend
        la freq dont le module est le + grand.
        Renvoie (freq, mod, phi)
        """
        f = np.fft.fftfreq(self.zero_padd * self.len, d=1/self.Fe)
        index = np.argmax(np.abs(self.complete_signal_fft)[0:(len(self.complete_signal_fft) + 1)//2])
        mod = 2 * np.abs(self.complete_signal_fft)[index][0] * np.sqrt(self.zero_padd) / np.sqrt(self.len)
        phi = np.angle(self.complete_signal_fft[index][0]) + np.pi/2
        return f[index], mod, phi

    def remove_frequency_from_signal(self, freq, mod, phi):
        t = np.arange(self.len).reshape(self.len, 1) / self.Fe
        self.complete_signal = self.complete_signal - (mod * np.sin(2 * np.pi * freq * t + phi)).reshape(self.len, 1)
        self.complete_signal_fft = np.fft.fft(self.complete_signal, axis= 0, n= self.zero_padd * self.len, norm='ortho')
    
    def remove_frequency_from_fft(self, freq, mod, phi):
        f = np.fft.fftfreq(self.zero_padd * self.len, d=1/self.Fe)
        #WIP
        return 


    def decode(self, show = False):
        """
        Ne va marcher que sur des char simples pour l'instant
        """
        if show : 
            self.show_signal()
            self.show_fft_signal()
        freq, mod, phi = self.get_biggest_frequency()
        #print(f'Current biggest module is {mod} at frequency {freq} and phase {phi}')
        i = 0
        while not (501 <= np.round(freq) <=526) and i < self.max_removals:
            self.remove_frequency_from_signal(freq, mod, phi)
            if show : 
                self.show_signal()
                self.show_fft_signal()
            freq, mod, phi = self.get_biggest_frequency()
            i+=1
            
            #print(f'Current biggest module is {mod} at frequency {freq} and phase {phi}')
        #print(i)
        if i < self.max_removals:
            #print(f'Char found is {self.get_symbol(np.round(freq))} or frequency {freq}, module {mod} and phase {phi}')
            return self.get_symbol(np.round(freq)), mod, i
        else:
            #print(f'No char found. Defaulting to a space')
            return " ", -1, i
    





def decode(s : np.ndarray):
    return Signal(s).decode(False)


if __name__ == '__main__':
    
    import IPython.display as ipd
    import soundfile as sf
    #########################################################################
    # Remarque : pour que le sf.read marche il faut être à la racine du git #
    #########################################################################
    x, Fe = sf.read('audio/mess_difficile.wav')
    #########################################################################
    # Remarque : pour que le sf.read marche il faut être à la racine du git #
    #########################################################################
    import time
    start = time.perf_counter()
    message = decode(x)
    end = time.perf_counter()
    print(f' Le message est {message} ')
    print(f'Le temps pour décoder a été de {end-start}s')

from scripts.filtrage import analyseNoteGwendal
from scripts.bruteforce import analyse
import soundfile as sf

def test1():
    analyseNoteGwendal()

def test2(file):
    x, Fe = sf.read(f"audio/{file}.wav")
    analyse(x, Fe)

if __name__ == "__main__":
    test2("gamme_demiTon_guitare")
    test2("love_me_SI101_drums")
    test2("love_me_SI101")
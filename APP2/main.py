from scripts.filtrage import analyseNoteGwendal
from scripts.bruteforce import analyseNoteRaphael

def test1():
    analyseNoteGwendal()

def test2(file):
    analyseNoteRaphael(file)

if __name__ == "__main__":
    #test1()
    test2("gamme_demiTon_guitare")
    test2("love_me_SI101_drums")
    test2("love_me_SI101")
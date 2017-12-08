from Features.FLyRBBuilder import FLyRBBuilder
from History import History
from MyParser import MyParser


def sanity():
    fLy = FLyRBBuilder()
    history = History("t1", "t1", ["w2", "w_five", "w_eight", "w_two"], 1)
    assert fLy.getFeatureVector(history, "CD").size == 0
    history = History("t1", "t1", ["w2", "W5ly", "w8", "w2"], 1)
    assert fLy.getFeatureVector(history, "RB").size == 1


def realData():
    p = MyParser('../train.wtag')
    fLy = FLyRBBuilder()
    sentence = [w for (w, t) in p.splitted[93]]
    history = History("RB", "VBG", sentence, 0)
    assert fLy.getFeatureVector(history, "adas").size == 0
    assert fLy.getFeatureVector(history, "RB").size == 1


if __name__ == "__main__":
    sanity()
    realData()

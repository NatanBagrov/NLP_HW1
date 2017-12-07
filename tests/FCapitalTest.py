from Features.FCapitalBuilder import FCapitalBuilder
from History import History
from MyParser import MyParser


def sanity():
    fCapital = FCapitalBuilder()
    history = History("t1", "t1", ["w2", "w5", "w8", "w2"], 1)
    assert fCapital.getFeatureVector(history, "t8").size == 0
    history = History("t1", "t1", ["w2", "W5", "w8", "w2"], 1)
    assert fCapital.getFeatureVector(history, "t8").size == 1


def realData():
    p = MyParser('../train.wtag')
    fCapital = FCapitalBuilder()
    firstSent = [w for (w, t) in p.splitted[0]]
    history = History("RB", "VBG", firstSent, 2)
    assert fCapital.getFeatureVector(history, "adas").size == 0
    history = History("RB", "VBG", firstSent, 1)
    assert fCapital.getFeatureVector(history, "RP").size == 1


if __name__ == "__main__":
    sanity()
    realData()

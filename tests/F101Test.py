from Features.F101Builder import F101Builder
from History import History
from MyParser import MyParser


def sanity():
    f101 = F101Builder()
    history = History("t5", "t2", ["endwithing", "w5", "w8", "w2"], 0)
    vec = f101.getFeatureVector(history, "VBG")
    assert vec.size == 1
    vec = f101.getFeatureVector(history, "t1")
    assert vec.size == 0


def realData():
    p = MyParser('../train.wtag')
    f101 = F101Builder()
    firstSent = [w for (w, t) in p.splitted[0]]
    history = History("t5", "t2", firstSent, 4)
    assert f101.getFeatureVector(history, "RB").size == 0
    assert f101.getFeatureVector(history, "VBG").size == 1


if __name__ == "__main__":
    realData()
    sanity()

from Features.F104Builder import F104Builder
from History import History
from MyParser import MyParser


def sanity():
    words = [("t1", "t2"), ("t4", "t3"), ("t3", "t7"), ("t4", "t1"), ("t1", "t9"), ("t1", "t1")]
    f104 = F104Builder(words)
    history = History("t1", "t1", ["w2", "w5", "w8", "w2"], 1)
    assert f104.getFeatureVector(history, "t8").size == 0
    assert f104.getFeatureVector(history, "t9").size == 1


def realData():
    p = MyParser('../train.wtag')
    words = p.getAllPairTagsCombinations()
    f104 = F104Builder(words)
    firstSent = [w for (w, t) in p.splitted[0]]
    history = History("RB", "VBG", firstSent, 3)
    assert f104.getFeatureVector(history, "bla").size == 0
    assert f104.getFeatureVector(history, "RP").size == 1


if __name__ == "__main__":
    sanity()
    realData()

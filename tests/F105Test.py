from Features.F104Builder import F104Builder
from Features.F105Builder import F105Builder
from History import History
from MyParser import MyParser


def sanity():
    words = ["t1", "t2", "t3" "t4", "t7", "t9"]
    f105 = F105Builder(words)
    history = History("t1", "t1", ["w2", "w5", "w8", "w2"], 1)
    assert f105.getFeatureVector(history, "t8").size == 0
    assert f105.getFeatureVector(history, "t9").size == 1


def realData():
    p = MyParser('../train.wtag')
    words = p.getUniqueTags()
    f105 = F105Builder(words)
    firstSent = [w for (w, t) in p.splitted[0]]
    history = History("RB", "VBG", firstSent, 3)
    assert f105.getFeatureVector(history, "adas").size == 0
    assert f105.getFeatureVector(history, "RP").size == 1


if __name__ == "__main__":
    sanity()
    realData()

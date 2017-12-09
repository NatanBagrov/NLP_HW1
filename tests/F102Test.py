from Features.F102Builder import F102Builder
from History import History
from MyParser import MyParser


def sanity():
    f102 = F102Builder()
    history = History("t5", "t2", ["endwithing", "w5", "prew8", "w2"], 2)
    vec = f102.getFeatureVector(history, "NN")
    assert vec.size == 1
    vec = f102.getFeatureVector(history, "t1")
    assert vec.size == 0


def realData():
    p = MyParser('../train.wtag')
    f102 = F102Builder()
    firstSent = [w for (w, t) in p.splitted[37]]
    history = History("t5", "t2", firstSent, 4)
    assert f102.getFeatureVector(history, "RB").size == 0
    assert f102.getFeatureVector(history, "NN").size == 1


if __name__ == "__main__":
    realData()
    sanity()

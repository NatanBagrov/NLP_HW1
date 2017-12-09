from Features.FAdjSuffixBuilder import FAdjSuffixBuilder
from History import History
from MyParser import MyParser


def sanity():
    fAble = FAdjSuffixBuilder()
    history = History("t1", "t1", ["w2", "w_five", "w_eight", "w_two"], 1)
    assert fAble.getFeatureVector(history, "CD").size == 0
    history = History("t1", "t1", ["w2", "W5able", "w8", "w2"], 1)
    assert fAble.getFeatureVector(history, "JJ").size == 1


def realData():
    p = MyParser('../train.wtag')
    fAble = FAdjSuffixBuilder()
    sentence = [w for (w, t) in p.splitted[175]]
    history = History("RB", "VBG", sentence, 0)
    assert fAble.getFeatureVector(history, "adas").size == 0
    history = History("RB", "VBG", sentence, 4)
    assert fAble.getFeatureVector(history, "JJ").size == 1


if __name__ == "__main__":
    sanity()
    realData()

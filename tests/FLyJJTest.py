from Features.FAbleBuilder import FAbleBuilder
import Features.FLyJJBuilder
from History import History
from MyParser import MyParser


def sanity():
    fLy = Features.FLyJJBuilder.FLyJJBuilder()
    history = History("t1", "t1", ["w2", "w_five", "w_eight", "w_two"], 1)
    assert fLy.getFeatureVector(history, "CD").size == 0
    history = History("t1", "t1", ["w2", "W5ly", "w8", "w2"], 1)
    assert fLy.getFeatureVector(history, "JJ").size == 1


def realData():
    p = MyParser('../train.wtag')
    fLy = Features.FLyJJBuilder.FLyJJBuilder()
    sentence = [w for (w, t) in p.splitted[175]]
    history = History("RB", "VBG", sentence, 0)
    assert fLy.getFeatureVector(history, "adas").size == 0
    history = History("RB", "VBG", sentence, 4)
    assert fLy.getFeatureVector(history, "JJ").size == 1


if __name__ == "__main__":
    sanity()
    realData()

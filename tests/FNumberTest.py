from Features.FNumberBuilder import FNumberBuilder
from History import History
from MyParser import MyParser


def sanity():
    fNumber = FNumberBuilder()
    history = History("t1", "t1", ["w2", "w_five", "w_eight", "w_two"], 1)
    assert fNumber.getFeatureVector(history, "t8").size == 0
    history = History("t1", "t1", ["w2", "W5", "w8", "w2"], 1)
    assert fNumber.getFeatureVector(history, "t8").size == 1


def realData():
    p = MyParser('../train.wtag')
    fNumber = FNumberBuilder()
    sentence = [w for (w, t) in p.splitted[18]]
    history = History("RB", "VBG", sentence, 4)
    assert fNumber.getFeatureVector(history, "adas").size == 0
    history = History("RB", "VBG", sentence, 2)
    assert fNumber.getFeatureVector(history, "RP").size == 1


if __name__ == "__main__":
    sanity()
    realData()

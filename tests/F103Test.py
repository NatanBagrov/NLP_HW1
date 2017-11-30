from Features.F103Builder import F103Builder
from History import History
from myParser import MyParser


def sanity():
    words = [("t1", "t2", "t3"), ("t4", "t3", "t1"), ("t3", "t7", "t8"), ("t4", "t1", "t2"), ("t1", "t2", "t1"),
             ("t1", "t1", "t2")]
    f103 = F103Builder(words)
    history = History("t1", "t2", ["w2", "w5", "w8", "w2"], 1)
    vec = f103.getFeatureVector(history, "t8")
    print(vec)
    vec1 = f103.getFeatureVector(history, "t1")
    print(vec1)


def realData():
    p = MyParser('../train.wtag')
    words = p.getAllThreeTagsCombinations()
    f103 = F103Builder(words)
    firstSent = [w for (w, t) in p.splitted[0]]
    history = History("RB", "VBG", firstSent, 3)
    print(f103.getFeatureVector(history, "bla").sum())
    print(f103.getFeatureVector(history, "RP").sum())


if __name__ == "__main__":
    sanity()
    realData()

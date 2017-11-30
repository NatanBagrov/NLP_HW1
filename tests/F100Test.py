from Features.F100Builder import F100Builder
from History import History
from myParser import MyParser


def sanity():
    words = [("w1", "t1"), ("w1", "t2"), ("w3", "t1"), ("w5", "t2"), ("w3", "t2")]
    f100 = F100Builder(words)
    history = History("t5", "t2", ["w2", "w5", "w8", "w2"], 1)
    vec = f100.getFeatureVector(history, "t2")
    print(vec)
    vec1 = f100.getFeatureVector(history, "t1")
    print(vec1)

def realData():
    p = MyParser('../train.wtag')
    words = p.getWordsWithTag()
    f100 = F100Builder(words)
    firstSent = [w for (w, t) in p.splitted[0]]
    history=History("t5","t2",firstSent,3)
    print(f100.getFeatureVector(history,"bla").sum())
    print(f100.getFeatureVector(history,"RB").sum())

if __name__=="__main__":
    realData()
    sanity()


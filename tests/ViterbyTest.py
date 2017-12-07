import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from History import History
from MLE import MLE
from MyParser import MyParser
from Viterbi import Viterbi


def basicTest():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1", "t2", "t3", "t5"], splitted, fb)
    v = np.ones(fb.size)
    sentence = ["w1", "w2", "w3", "w2"]
    sentence2 = ["w4", "w3"]
    vit = Viterbi(mle, mle.allTags, v)
    vit.inference(sentence)
    #vit.inference(sentence2)



def trainTest():
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(parser.getUniqueTags(), splitted, fb)
    v = np.ones(fb.size)
    splitted = splitted[0:1]
    sentences = list(map(lambda tuples: [t[0] for t in tuples], splitted))
    expected_tags = list(map(lambda tuples: [t[1] for t in tuples], splitted))
    vit = Viterbi(mle, mle.allTags, v)
    res = 0
    for s,expected in zip(sentences,expected_tags):
        tags = vit.inference(s)
        e = np.array([hash(x) for x in expected])
        t = np.array([hash(x) for x in tags])
        np_sum = np.sum(e == t)
        res = res + np_sum
    print(res)

def veriftPi():
   pass


if __name__ == "__main__":
    #trainTest()
    start = time.time()
    basicTest()
    end = time.time()
    #print((end - start) / 60)
    veriftPi()


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
    mle = MLE(["t1","t2","t3","t5"],splitted,fb)
    v = np.ones(fb.size)
    sentence = ["w1", "w2", "w3", "w2"]
    vit = Viterbi(mle,mle.allTags,v)
    vit.inference(sentence)

if __name__=="__main__":
    start = time.time()
    basicTest()
    end = time.time()
    print((end - start) / 60)



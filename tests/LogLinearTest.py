import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from History import History
from MLE import MLE
from MyParser import MyParser


def basicTest():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1","t2","t3","t5"],splitted,fb)
    v = np.ones(fb.size)
    history = History("t1", "t2", ["w1", "w2", "w3", "w2"], 2)
    res = mle.p(history,"t3",v)
    print(res)

if __name__=="__main__":
    start = time.time()
    basicTest()
    end = time.time()
    print((end - start) / 60)



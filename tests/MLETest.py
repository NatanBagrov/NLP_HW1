import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser


def basicTest():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1","t2","t3","t5"],splitted,fb)
    v = np.ones(fb.size)
    print(mle.calculate(v))
    print(mle.calculateGradient(v))

def realDataTest():

    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, fb)
    v = np.ones(fb.size)
    print(mle.calculate(v))
    f = open("train_gradient.txt","w")
    array = mle.calculateGradient(v)
    np.savetxt('train_gradient.txt',array)

if __name__=="__main__":
    start = time.time()
    #basicTest()
    end = time.time()
    print((end - start)/60)
    start = time.time()
    realDataTest()
    end = time.time()
    print((end - start)/60)
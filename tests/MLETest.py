import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser

def calcTupleTestBasic():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1", "t2", "t3", "t5"], splitted, fb)
    v = np.ones(fb.size)
    res = mle.calcTuple(v)
    print(res)

def calcTupleTestRealData():
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    start = time.time()
    mle = MLE(tags, splitted, fb)
    end = time.time()
    print("End of preprocessing, took: ", end - start)
    v = np.ones(fb.size)
    start = time.time()
    f = open("train_gradientTuple.txt", "w")
    lv,grad = mle.calcTuple(v)
    print("L(V) = ", lv)
    print(grad)
    np.savetxt('train_gradientTuple.txt', grad)
    end = time.time()
    print("calcTuple took: ",end - start, " seconds")
    truth = np.loadtxt("train_gradient.txt")
    current = np.loadtxt("train_gradientTuple.txt")
    dist = np.linalg.norm(truth - current)
    assert dist < 0.0001

def basicTest():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1","t2","t3","t5"],splitted,fb)
    v = np.ones(fb.size)
    res = mle.calculateGradient(v)
    print(res)

def realDataTest():

    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    start = time.time()
    mle = MLE(tags, splitted, fb)
    end = time.time()
    print("End of preprocessing, took: ", end-start)
    v = np.ones(fb.size)
    start = time.time()
    print(mle.calculate(v))
    end = time.time()
    print("calcV took: " + str((end - start) / 60))
    start = time.time()
    f = open("train_gradient2.txt","w")
    array = mle.calculateGradient(v)
    np.savetxt('train_gradient2.txt',array)
    end = time.time()
    print("calcGrad took: " + str((end - start) / 60))
    truth = np.loadtxt("train_gradient.txt")
    current = np.loadtxt("train_gradient2.txt")
    dist = np.linalg.norm(truth - current)
    print(dist)


if __name__=="__main__":
    calcTupleTestBasic();
    start = time.time()
    #basicTest()
    end = time.time()
    print((end - start)/60)
    start = time.time()
    #realDataTest()
    end = time.time()
    print((end - start)/60)
    start = time.time()
    print("Calculating tuple using real data...")
    calcTupleTestRealData()
    end = time.time()
    print((end - start) / 60)



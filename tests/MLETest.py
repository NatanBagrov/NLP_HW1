import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser

def calcTupleTestBasic():
    parser = MyParser("MLE_db.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    mle = MLE(["t1", "t2", "t3", "t5"], splitted, fb)
    v = np.zeros(fb.size)
    res = mle.calcTuple(v)
    print(res)
    best_v = mle.findBestV()
    print(best_v)
    res1 = mle.calcTuple(best_v)
    print(res1)

def calcTupleTestRealData():
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    # fb = BasicFeatureVectorBuilder(parser,0)
    fb = ComplexFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    start = time.time()
    mle = MLE(tags, splitted, fb,0,"tmp1234.txt")
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
    array = mle.calculateGradient(v)
    np.savetxt('train_gradient2.txt',array)
    end = time.time()
    print("calcGrad took: " + str((end - start) / 60))
    truth = np.loadtxt("train_gradient.txt")
    current = np.loadtxt("train_gradient2.txt")
    dist = np.linalg.norm(truth - current)
    print(dist)
    best_v = mle.findBestV()
    print(best_v)


def TRAIN():
    print("Training: ")
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, fb)
    best_v = mle.findBestV(np.loadtxt("opt_v.txt"))
    print(best_v)


if __name__=="__main__":

    #TRAIN()
    calcTupleTestRealData()
    #calcTupleTestBasic()
    #start = time.time()
    #basicTest()
    #end = time.time()
    #print((end - start)/60)
    #start = time.time()
    #realDataTest()
    #end = time.time()
    #print((end - start)/60)
    # start = time.time()
    # print("Calculating tuple using real data...")
    # calcTupleTestRealData()
    # end = time.time()
    # print((end - start) / 60)



import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser

def TRAIN_Complex():
    lambdas = [0, 0.05, 2]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    cfb = ComplexFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    for lmbda in lambdas:
        print("Current lambda:", str(lmbda))
        start = time.time()
        tmpFile = 'complex_opt_v_lambda_' + str(lmbda).replace('.','_') + '.txt'
        mle = MLE(tags, splitted, cfb)
        best_v = mle.findBestV(np.zeros(cfb.size),lmbda, tmpFile)
        resFile = 'finish_' + tmpFile
        np.savetxt(resFile,best_v.x)
        print("Training lambda: ", str(lmbda), " took: ", (time.time() - start)/60, "minutes")


def TRAIN_Basic():
    lambdas = [0.001, 0.005, 0.007, 0.01, 0.012, 0.015, 0.017]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    cfb = BasicFeatureVectorBuilder(parser,0)
    tags = parser.getUniqueTags()
    for lmbda in lambdas:
        print("Current lambda:", str(lmbda))
        start = time.time()
        tmpFile = 'basic_opt_v_lambda_' + str(lmbda).replace('.','_') + '.txt'
        mle = MLE(tags, splitted, cfb,)
        best_v = mle.findBestV(np.zeros(cfb.size),lmbda, tmpFile)
        resFile = 'finish_' + tmpFile
        np.savetxt(resFile,best_v.x)
        print("Training lambda: ", str(lmbda), " took: ", (time.time() - start)/60, "minutes")

if __name__=="__main__":
    start = time.time()
    TRAIN_Basic()
    # TRAIN_Complex()
    print('Training all lambdas took ', (time.time() - start)/60, 'minutes')
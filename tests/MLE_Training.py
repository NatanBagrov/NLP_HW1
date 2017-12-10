import numpy as np
import time

from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser

def TRAIN():
    lambdas = [0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
               0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.5, 1, 1.5, 2]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    cfb = ComplexFeatureVectorBuilder(parser)
    tags = parser.getUniqueTags()
    for l in lambdas:
        print("Current lambda:", str(l))
        start = time.time()
        tmpFile = 'complex_opt_v_lambda_' + str(l).replace('.','_') + '.txt'
        mle = MLE(tags, splitted, cfb,l, tmpFile)
        best_v = mle.findBestV(np.zeros(cfb.size))
        resFile = 'finish_' + tmpFile
        np.savetxt(resFile,best_v.x)
        print("Training lambda: ", str(l), " took: ", (time.time() - start)/60, "minutes")

if __name__=="__main__":
    start = time.time()
    TRAIN()
    print('Training all lambdas took ', (time.time() - start)/60, 'minutes')
import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser


def fit_complex_model():
    lambdas = [0, 0.05, 2]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    cfb = ComplexFeatureVectorBuilder(parser, parser, True)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, cfb)
    fit_model_aux(mle, "complex", lambdas, 400)


def fit_basic_model():
    lambdas = [0.001, 0.005, 0.007, 0.01, 0.012, 0.015, 0.017]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    basicFeatureBuilder = BasicFeatureVectorBuilder(parser, 0)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, basicFeatureBuilder)
    fit_model_aux(mle, "basic", lambdas, 400)


def fit_model_aux(mle: MLE, prefix_name, lambdas, iterationsNum, initv=None):
    print("Starting training with lambdas:", lambdas, "on:", prefix_name)
    if initv is None:
        print("Training with initial vector of zeros")
        v = np.zeros(mle.featureBuilder.size)
    else:
        print("Will continue training given init vector")
        v = initv
    for lmbda in lambdas:
        print("Current lambda:", str(lmbda))
        start = time.time()
        tmpFile = prefix_name + '_opt_v_lambda_' + str(lmbda).replace('.', '_') + '.txt'
        best_v = mle.findBestV(v, lmbda, tmpFile, iterationsNum)
        resFile = 'finish_' + tmpFile
        np.savetxt(resFile, best_v.x)
        print("Training lambda: ", str(lmbda), " took: ", (time.time() - start) / 60, "minutes")
        print("######################################################")


if __name__ == "__main__":
    fit_basic_model()
    # fit_complex_model()

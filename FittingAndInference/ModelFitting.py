import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser


def fit_complex_model(continueTraining):
    v = None
    if continueTraining:
        v = np.loadtxt("finish_complex_opt_v_lambda_0_007.txt")
    lambdas = [0.007]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    cfb = ComplexFeatureVectorBuilder(parser, False)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, cfb)
    fit_model_aux(mle, "complex", lambdas, 300, v)


def fit_basic_model(continueTraining):
    v = None
    if continueTraining:
        v = np.loadtxt("finish_basic_opt_v_lambda_0_007.txt")
    lambdas = [0.007]
    parser = MyParser("../train.wtag")
    splitted = parser.splitted
    basicFeatureBuilder = BasicFeatureVectorBuilder(parser, 0)
    tags = parser.getUniqueTags()
    mle = MLE(tags, splitted, basicFeatureBuilder)
    fit_model_aux(mle, "basic", lambdas, 550, v)


def fit_model_aux(mle: MLE, prefix_name, lambdas, iterationsNum, initv=None):
    print("Starting training with lambdas:", lambdas, "on:", prefix_name)
    if initv is None:
        print("Training with initial vector of zeros")
        v = np.zeros(mle.featureBuilder.size)
    else:
        print("Will continue training given init vector")
        v = initv
        print(v)
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
    # fit_basic_model(True)
    fit_complex_model(False)

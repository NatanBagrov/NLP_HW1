import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser
from Viterbi import Viterbi



def findBestLambda(testFile):
    train_parser = MyParser("../train.wtag")
    seenSentencesToTagsDict = train_parser.getSeenWordsToTagsDict()
    parser = MyParser(testFile)
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser,0)
    import os
    prefixed = [filename for filename in os.listdir('.') if filename.startswith('finish_basic_opt_v_')]
    prefixed.sort()
    print(prefixed)
    lambdas = [0.5,0.02, 0.025]
    results = []
    for file,lmbda in zip(prefixed,lambdas):
        mle = MLE(parser.getUniqueTags(), splitted, fb, lmbda,'../tmp1234.txt')
        v = np.loadtxt(file)
        splitted = splitted[309:]
        sentences = list(map(lambda tuples: [t[0] for t in tuples], splitted))
        expected_tags = list(map(lambda tuples: [t[1] for t in tuples], splitted))
        vit = Viterbi(mle, mle.allTags, v, seenSentencesToTagsDict)
        total_res = 0
        words_count = 0
        total_time = 0
        accuracy = 0
        for s,expected,idx in zip(sentences,expected_tags,range(0,len(splitted))):
            curr_word_len = len(s)
            words_count = words_count + curr_word_len
            start = time.time()
            tags = vit.inference(s)

            res_file = open("test_wtag_results_lambda_" + str(lmbda).replace('.','_') + ".txt",'a')
            for item in tags:
                res_file.write("%s " % item)
            res_file.write("\n")
            res_file.close()

            exp_file = open("test_wtag_expected_lambda_" + str(lmbda).replace('.','_') + ".txt", 'a')
            for item in expected:
                exp_file.write("%s " % item)
            exp_file.write("\n")
            exp_file.close()

            stop = time.time()
            e = np.array([hash(x) for x in expected])
            t = np.array([hash(x) for x in tags])
            current_correct = np.sum(e == t)
            print("---------------------")
            print("Inference for sentence# ", idx, " took: ", stop - start, " seconds")
            total_time = total_time + (stop-start)
            print("Current sentence accuracy: ", current_correct, " of: ", curr_word_len)
            total_res = total_res + current_correct
            accuracy = (100 * total_res) / words_count
            print("Total sentence accuracy: ", total_res, " of: ", words_count, "=", accuracy, "%")
            print("Total time for ", idx, " sentences: ", (total_time / 60), " minutes")
        results = results + [accuracy]
    res = np.array(results)
    best_lambda_idx = res.argmax()
    print(results)
    print("Best index:", best_lambda_idx)

if __name__ == "__main__":
    findBestLambda("../test.wtag")

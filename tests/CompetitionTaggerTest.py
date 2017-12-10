import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from History import History
from MLE import MLE
from MyParser import MyParser
from Viterbi import Viterbi

def train():
    train_parser = MyParser("../train.wtag")
    seenSentencesToTagsDict = train_parser.getSeenWordsToTagsDict()
    parser = MyParser("../comp748.wtag")
    splitted = parser.splitted
    fb = BasicFeatureVectorBuilder(parser,0)
    mle = MLE(parser.getUniqueTags(), splitted, fb)
    v = np.loadtxt("opt_v_3.txt")
    sentences = list(map(lambda tuples: [t[0] for t in tuples], splitted))
    expected_tags = list(map(lambda tuples: [t[1] for t in tuples], splitted))
    seenSentencesToTagsDict = parser.getSeenWordsToTagsDict()
    vit = Viterbi(mle, mle.allTags, v, seenSentencesToTagsDict)
    total_res = 0
    words_count = 0
    total_time = 0
    for s,expected,idx in zip(sentences,expected_tags,range(0,len(splitted))):
        curr_word_len = len(s)
        words_count = words_count + curr_word_len
        start = time.time()
        tags = vit.inference(s)

        res_file = open("test_wtag748_results.txt",'a')
        for item in tags:
            res_file.write("%s " % item)
        res_file.write("\n")
        res_file.close()

        exp_file = open("test_wtag748_expected.txt", 'a')
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
        print("Total sentence accuracy: ", total_res, " of: ", words_count, "=", (100*total_res)/words_count, "%")
        print("Total time for ", idx, " sentences: ", (total_time / 60), " minutes")

if __name__ == "__main__":
    train()

import numpy as np
import time
import os
from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MLE import MLE
from MyParser import MyParser
from Viterbi import Viterbi


def infer_basic(fileToInfer):
    infer_prepare_params("basic", fileToInfer)

def infer_complex(fileToInfer):
    infer_prepare_params("complex", fileToInfer)

def infer_prepare_params(basic_or_complex, fileToInfer):
    train_parser = MyParser("../train.wtag")
    seenWordsToTagsDict = train_parser.getSeenWordsToTagsDict()
    fb,filePrefix = None, None
    if basic_or_complex == 'basic':
        fb = BasicFeatureVectorBuilder(train_parser, 0)
        filePrefix = 'finish_basic_opt_v_'
    elif basic_or_complex == 'complex':
        fb = ComplexFeatureVectorBuilder(train_parser, False)
        filePrefix = 'finish_complex_opt_v_'
    else:
        assert (False)
    fn = str(fileToInfer).replace('.','').replace('/','')
    parser = MyParser(fileToInfer)
    splitted = parser.splitted
    mle = MLE(parser.getUniqueTags(), splitted, fb)

    prefixed = [filename for filename in os.listdir('.') if filename.startswith(filePrefix)]
    prefixed.sort()
    print(prefixed)
    results = []

    for v_file in prefixed:
        v = np.loadtxt(v_file)
        vit = Viterbi(mle, mle.allTags, v, seenWordsToTagsDict)
        res_file = open(fn+"_results_" + v_file, 'w')
        exp_file = open(fn+"_expected_" + v_file, 'w')
        accuracy = infer_aux(exp_file, res_file, v_file, splitted, vit)
        res_file.close()
        exp_file.close()
        results = results + [accuracy]
    infer_aux_results(prefixed, results, fileToInfer, fn)

def infer_aux_results(prefixed, results, fileToInfer, fn):
    summary_file = open("summary_inference.txt", 'a')
    for f, r in zip(prefixed, results):
        s = "Results for " + fileToInfer + " " + fn + " " + "sentences, with params: " + f + " is: " + str(r) + "\n"
        summary_file.write(s)
    summary_file.write(".\n\n")
    summary_file.close()
    res = np.array(results)
    best_lambda_idx = res.argmax()
    print(results)
    print("Best index:", best_lambda_idx)

def infer_aux(exp_file, res_file, v_file, splitted, vit):
    total_res, words_count, total_time, accuracy = 0, 0, 0, 0
    sentences = list(map(lambda tuples: [t[0] for t in tuples], splitted))
    expected_tags = list(map(lambda tuples: [t[1] for t in tuples], splitted))
    for s, expected, idx in zip(sentences, expected_tags, range(0, len(splitted))):
        start = time.time()
        curr_word_len = len(s)
        words_count = words_count + curr_word_len
        tags = vit.inference(s)
        stop = time.time()

        for item in tags:
            res_file.write("%s " % item)
        res_file.write("\n")
        for item in expected:
            exp_file.write("%s " % item)
        exp_file.write("\n")

        e = np.array([hash(x) for x in expected])
        t = np.array([hash(x) for x in tags])
        current_correct = np.sum(e == t)
        print("---------------------")
        print("Inference for sentence# ", idx, " took: ", stop - start, " seconds")
        total_time = total_time + (stop - start)
        print("Current sentence accuracy: ", current_correct, " of: ", curr_word_len, "using: ", v_file)
        total_res = total_res + current_correct
        accuracy = (100 * total_res) / words_count
        print("Total sentence accuracy: ", total_res, " of: ", words_count, "=", accuracy, "%")
        print("Total time for ", idx, " sentences: ", (total_time / 60), " minutes")
    return accuracy


if __name__ == "__main__":
    infer_basic("../test.wtag")
    infer_basic("../comp748.wtag")
    infer_complex("../test.wtag")
    infer_complex("../comp748.wtag")

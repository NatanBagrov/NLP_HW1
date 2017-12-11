import multiprocessing
from scipy import optimize
import numpy as np
import time

from Features.FeatureBuilderBase import FeatureBuilderBase
from History import History


class MLE():
    sumfiyi = None
    lmbda = 0
    v = {}
    tmpVFile = 'opt_v_tmp.txt'

    def __init__(self, allTags, splitted, featureBuilder: FeatureBuilderBase) -> None:
        super().__init__()
        self.allTags = allTags
        self.splitted = splitted
        self.featureBuilder = featureBuilder
        self.sumfiyi = np.zeros(self.featureBuilder.size)
        self.preprocess()

    def p(self, history, tag, v):
        feature = self.featureBuilder.getFeatureVector(history, tag)
        numerator = np.math.exp(np.sum(v[feature]))
        fs = [self.featureBuilder.getFeatureVector(history, t) for t in self.allTags]
        np_sums = np.array([np.sum(v[x]) for x in fs])
        np_exp_nominators = np.exp(np_sums)
        np_exp_sum = np.sum(np_exp_nominators)
        return numerator / np_exp_sum

    def p_forTags(self,history,tag,v,tags):
        feature = self.featureBuilder.getFeatureVector(history, tag)
        numerator = np.math.exp(np.sum(v[feature]))
        fs = [self.featureBuilder.getFeatureVector(history, t) for t in tags]
        np_sums = np.array([np.sum(v[x]) for x in fs])
        np_exp_nominators = np.exp(np_sums)
        np_exp_sum = np.sum(np_exp_nominators)
        return numerator / np_exp_sum

    def preprocess(self):
        for line in self.splitted:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                f = self.featureBuilder.getFeatureVector(history, tags[index + 2])
                self.sumfiyi[f] += 1

    def calcTuple(self, v):
        np.savetxt(self.tmpVFile, v)
        self.v = v
        poolSize = 7
        splitted = self.slice_list(list(range(0, len(self.splitted))), poolSize)
        splitted = list(filter(lambda x: len(x) > 0, splitted))
        se = [(l[0], l[-1]) for l in splitted]
        pool = multiprocessing.Pool(poolSize)
        res = pool.imap(self.calcTupleMP, se)
        x = np.array([np.array(x) for x in res if not x is None])
        grads = np.array([np.array(xx[0]) for xx in x])
        lv = np.array([np.array(xx[1]) for xx in x])
        # grads, lv = self.calcTupleMP((0, len(self.splitted)))
        grads = self.sumfiyi - np.sum(grads, axis=0)
        grads_regularizator = self.v * self.lmbda
        grads = grads - grads_regularizator
        lv = np.sum(lv)
        lv_regularizator = np.inner(self.v, self.v) * (self.lmbda / 2)
        lv = lv - lv_regularizator
        pool.close()
        pool.join()
        return -lv, -grads

    def calcTupleMP(self, indices):
        v = self.v
        globalScore, linearScore, score = 0, 0, 0
        mySumfiyi = np.zeros(self.featureBuilder.size)
        for line in self.splitted[indices[0]:indices[1] + 1]:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                f = self.featureBuilder.getFeatureVector(history, tags[index + 2])
                mySumfiyi[f] += 1
        linearScore = np.inner(mySumfiyi, v)
        weighted_sum = np.zeros(self.featureBuilder.size)
        for line in self.splitted[indices[0]:indices[1] + 1]:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                fs = [self.featureBuilder.getFeatureVector(history, t) for t in self.allTags]
                np_sums = np.array([np.sum(v[x]) for x in fs])
                np_exp_nominators = np.exp(np_sums)
                np_exp_sum = np.sum(np_exp_nominators)
                globalScore = globalScore + np.math.log(np_exp_sum)
                np_ps_of_ytag = np_exp_nominators / np_exp_sum
                np_features_mult_probs = np.zeros(self.featureBuilder.size)
                for i in range(0, len(self.allTags)):
                    np_features_mult_probs[fs[i]] += np_ps_of_ytag[i]
                weighted_sum = weighted_sum + np_features_mult_probs
        return weighted_sum, [linearScore - globalScore]

    def slice_list(self, input, size):
        input_size = len(input)
        slice_size = input_size // size
        remain = input_size % size
        result = []
        iterator = iter(input)
        for i in range(size):
            result.append([])
            for j in range(slice_size):
                result[i].append(next(iterator))
            if remain:
                result[i].append(next(iterator))
                remain -= 1
        return result

    def findBestV(self, initV, lmbda, tmpFile):
        self.lmbda = lmbda
        self.tmpVFile = tmpFile
        v = optimize.minimize(self.calcTuple, initV,
                              method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': 400})
        return v

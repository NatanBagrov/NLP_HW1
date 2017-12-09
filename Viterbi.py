import multiprocessing

import numpy as np
from itertools import product

import time

from History import History
from MLE import MLE


class Viterbi:
    mle = None
    tags = None
    v = None
    tagsToIdxDict = None
    idxToTagsDict = None
    seenWordsToTagsDict = None

    def __init__(self, mle: MLE, allTags, v, seenWordsToTagsDict) -> None:
        super().__init__()
        self.k = None
        self.mle = mle
        self.tags = allTags
        self.v = v
        self.tagsToIdxDict = {}
        self.idxToTagsDict = {}
        self.tagsNum = len(self.tags)
        self.seenWordsToTagsDict = seenWordsToTagsDict
        for tag, idx in zip(self.tags, range(0, self.tagsNum)):
            self.tagsToIdxDict[tag] = idx
            self.idxToTagsDict[idx] = tag
        self.tagsToIdxDict['*'] = self.tagsNum
        self.idxToTagsDict[self.tagsNum] = '*'
        input = [self.tags, self.tags]
        self.relevantTagTuples = list(product(*input))

    def inference(self, sentence):
        self.inferenceSetUp(sentence)
        self.inferenceFirstIteration(sentence)
        if len(sentence) > 1:
            self.inferenceSecondIteration(sentence)
        poolSize = 3
        for self.k in range(3, len(sentence) + 1):
            start = time.time()
            splitted = self.slice_list(list(range(0, len(self.relevantTagTuples))), poolSize)
            splitted = list(filter(lambda x: len(x) > 0, splitted))
            se = [(l[0], l[-1]) for l in splitted]
            pool = multiprocessing.Pool(poolSize)
            res = pool.imap(self.viterbiLoop, se)
            x = np.array([np.array(x) for x in res if not x is None])
            # work...
            localPi = np.array([np.array(xx[0]) for xx in x])
            localBp = np.array([np.array(xx[1]) for xx in x])
            localPi = np.sum(localPi, axis=0)
            localBp = np.sum(localBp, axis=0)
            self.pi[self.k] = localPi
            self.bp[self.k] = localBp
            pool.close()
            pool.join()
            stop = time.time()
            print("k= ", self.k, " took ", stop - start, " seconds")
        tagsList = self.inferenceLastIteration(sentence)
        return tagsList

    def inferenceLastIteration(self, sentence):
        p = self.pi[len(sentence)]
        t1, t = np.unravel_index(p.argmax(), p.shape)
        tagsList = [t, t1]
        tk1, tk2 = t1, t
        loopSize = len(sentence) - 1
        for k in reversed(range(1, loopSize)):
            tk = self.bp[k + 2, tk1, tk2]
            tagsList = tagsList + [tk]
            tk1, tk2 = tk, tk1
        tagsList = list(map(lambda x: self.idxToTagsDict[x], tagsList))
        tagsList.reverse()
        return tagsList

    def inferenceSecondIteration(self, sentence):
        for tagU, tagV in self.relevantTagTuples:
            if sentence[0] in self.seenWordsToTagsDict:
                if tagU not in self.seenWordsToTagsDict[sentence[0]]:
                    continue
            if sentence[1] in self.seenWordsToTagsDict:
                if tagV not in self.seenWordsToTagsDict[sentence[1]]:
                    continue
            history = History('*', tagU, sentence, 1)
            self.pi[2, self.tagsToIdxDict[tagU], self.tagsToIdxDict[tagV]] = \
                self.pi[1, self.tagsToIdxDict['*'], self.tagsToIdxDict[tagU]] \
                * self.mle.p_forTags(history, tagV, self.v, self.seenWordsToTagsDict[sentence[1]])
            self.bp[2, self.tagsToIdxDict[tagU], self.tagsToIdxDict[tagV]] = self.tagsToIdxDict['*']

    def inferenceFirstIteration(self, sentence):
        for tagV in self.tags:
            if sentence[0] in self.seenWordsToTagsDict:
                if tagV not in self.seenWordsToTagsDict[sentence[0]]:
                    continue
            history = History('*', '*', sentence, 0)
            self.pi[1, self.tagsToIdxDict['*'], self.tagsToIdxDict[tagV]] = \
                self.mle.p_forTags(history, tagV, self.v,self.seenWordsToTagsDict[sentence[0]])
            self.bp[1, self.tagsToIdxDict['*'], self.tagsToIdxDict[tagV]] = self.tagsToIdxDict['*']

    def inferenceSetUp(self, sentence):
        self.sentence = sentence
        self.pi = np.empty((len(sentence) + 1, len(self.tagsToIdxDict), len(self.tagsToIdxDict)))
        self.bp = np.empty((len(sentence) + 1, len(self.tagsToIdxDict), len(self.tagsToIdxDict)), dtype=int)
        self.pi[:, :, :] = 0
        self.bp[:, :, :] = 0
        self.pi[0, len(self.tagsToIdxDict) - 1, len(self.tagsToIdxDict) - 1] = 1
        self.bp[0, len(self.tagsToIdxDict) - 1, len(self.tagsToIdxDict) - 1] = self.tagsNum

    def viterbiLoop(self, se):
        allTagsList, bp, k, pi, sentence = self.relevantTagTuples, self.bp, self.k, self.pi, self.sentence
        myPi = np.zeros((len(self.tagsToIdxDict), len(self.tagsToIdxDict)))
        myBp = np.zeros((len(self.tagsToIdxDict), len(self.tagsToIdxDict)), dtype=int)
        for tagU, tagV in allTagsList[se[0]:se[1] + 1]:
            if sentence[k - 1] in self.seenWordsToTagsDict:
                if tagV not in self.seenWordsToTagsDict[sentence[k - 1]]:
                    continue
            if sentence[k - 2] in self.seenWordsToTagsDict:
                if tagU not in self.seenWordsToTagsDict[sentence[k - 2]]:
                    continue
            tmpMax = -1
            tmpMaxT = self.tagsNum
            for tagT in self.tags:
                if sentence[k - 3] in self.seenWordsToTagsDict:
                    if tagT not in self.seenWordsToTagsDict[sentence[k - 3]]:
                        continue
                history = History(tagT, tagU, sentence, k - 1)
                mleRes = self.mle.p_forTags(history, tagV, self.v, self.seenWordsToTagsDict[sentence[k - 1]])
                tmpRes = pi[k - 1, self.tagsToIdxDict[tagT], self.tagsToIdxDict[tagU]] * mleRes
                if tmpRes > tmpMax:
                    tmpMax, tmpMaxT = tmpRes, tagT
            myPi[self.tagsToIdxDict[tagU], self.tagsToIdxDict[tagV]] = tmpMax
            myBp[self.tagsToIdxDict[tagU], self.tagsToIdxDict[tagV]] = self.tagsToIdxDict[tmpMaxT]
        return (myPi, myBp)

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

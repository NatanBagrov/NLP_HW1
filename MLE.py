import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from History import History


class MLE():
    historyAndTag2Score = {}

    def __init__(self, allTags, splitted, featureBuilder: FeatureBuilderBase) -> None:
        super().__init__()
        self.allTags = allTags
        self.splitted = splitted
        self.featureBuilder = featureBuilder
        # self.preprocess()

    def preprocess(self):
        for line in self.splitted:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                self.addHistoryWithTagsToDict(history)

    def addHistoryWithTagsToDict(self, history):
        for tag in self.allTags:
            f = self.featureBuilder.getFeatureVector(history, tag)
            self.historyAndTag2Score[(history, tag)] = f

    def calculate(self, v):
        globalScore, linearScore, score = 0, 0, 0
        for line in self.splitted:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                for tag in self.allTags:
                    f = self.featureBuilder.getFeatureVector(history, tag)  # self.historyAndTag2Score[(history,tag)]
                    np_sum = np.sum(v[f])
                    expo = np.math.exp(np_sum)
                    if tag == tags[index + 2]:
                        linearScore = linearScore + np_sum
                    score = score + expo
                globalScore = globalScore + np.math.log(score)
                score = 0
        return (linearScore, globalScore)

    def calculateL(self, v):
        (linearScore, globalScore) = self.calculate(v)
        return linearScore - globalScore

    def calculateGradient(self, v):
        weighted_sum = np.zeros(self.featureBuilder.size)
        weighted_linear = np.zeros(self.featureBuilder.size)
        for line in self.splitted:
            sentence = [w[0] for w in line]
            tags = ["*", "*"] + [w[1] for w in line]
            for (t2, t1, index) in zip(tags[:], tags[1:], range(0, len(sentence))):
                history = History(t2, t1, sentence, index)
                exp_sum = 0
                f_log = []  # f(x^(i),y')
                for tag in self.allTags:
                    f = self.featureBuilder.getFeatureVector(history, tag)  # self.historyAndTag2Score[(history,tag)]
                    f_log.append(f)
                    np_sum = np.sum(v[f])
                    expo = np.math.exp(np_sum)
                    exp_sum = exp_sum + expo
                    if tag == tags[index + 2]:
                        tmp = np.zeros(self.featureBuilder.size)
                        tmp[f] = 1
                        weighted_linear = weighted_linear + tmp
                for featurevec in f_log:
                    nominator = np.math.exp(np.sum(v[featurevec]))
                    res = nominator / exp_sum
                    tmp = np.zeros(self.featureBuilder.size)
                    tmp[featurevec] = 1
                    tmp = tmp * res
                    weighted_sum = weighted_sum + tmp
        return weighted_linear - weighted_sum

import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class CapsFeatureBuilder(FeatureBuilderBase):
    d_train = {}
    d_inference = {}

    def __init__(self, offset, parser: MyParser) -> None:
        self.d_train = {}
        self.d_inference = {}
        caps_tuple = parser.getAllTagsForCaps()

        caps_len = len(caps_tuple)
        for (w, t), i in zip(caps_tuple, range(0, caps_len)):
            if t not in self.d_inference:
                self.d_inference[t] = i + offset
            self.d_train[(w, t)] = self.d_inference[t]

        super().__init__(len(self.d_inference), offset)

    def getFeatureVectorTrain(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        tpl = (current_word, tag)
        if tpl in self.d_train:
            return np.array([self.d_train[tpl]])
        return np.array([])

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag not in self.d_inference:
            return np.array([])
        if not str(current_word).islower():
            return np.array([self.d_inference[tag]])
        return np.array([])

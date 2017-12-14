import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class DigitNumberFeatureBuilder(FeatureBuilderBase):
    d_train = {}
    d_inference = {}

    def __init__(self, parser: MyParser, offset) -> None:
        self.d_train = {}
        self.d_inference = {}
        digit_letters_tuple = parser.getAllTagsForDigitLetters()

        digit_letters_len = len(digit_letters_tuple)
        for (w, t), i in zip(digit_letters_tuple, range(0, digit_letters_len)):
            if t not in self.d_inference:
                self.d_inference[t] = len(self.d_inference) + offset
            self.d_train[(w,t)] = self.d_inference[t]

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
        if any(i.isdigit() for i in str(current_word)):
                return np.array([self.d_inference[tag]])
        return np.array([])

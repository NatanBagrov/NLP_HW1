import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class DigitWordFeatureBuilder(FeatureBuilderBase):
    d_train = {}
    d_inference = {}
    digits = None

    def __init__(self, parser: MyParser, offset) -> None:
        self.d_train = {}
        self.d_inference = {}
        self.digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        digit_triplet = parser.getAllTagsForLettersNumbers(self.digits)
        # caps_tuple = parser.getAllTagsForCaps()
        # digit_letters_tuple = parser.getAllTagsForLettersNumbers(self.digits)

        digit_len = len(digit_triplet)
        for (w, x, t), i in zip(digit_triplet, range(0, digit_len)):
            if (x, t) not in self.d_inference:
                self.d_inference[(x, t)] = i + offset
            self.d_train[(w, t)] = self.d_inference[(x, t)]

        super().__init__(len(self.d_inference), offset)

    def getFeatureVectorTrain(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        tpl = (current_word, tag)
        if tpl in self.d_train:
            return np.array([self.d_train[tpl]])
        return np.array([])

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        for digit in self.digits:
            if str(current_word).startswith(digit):
                tpl = (digit, tag)
                if tpl in self.d_inference:
                    return np.array([self.d_inference[tpl]])
                return np.array([])
        return np.array([])

import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FVerbSuffixBuilder(FeatureBuilderBase):
    suffix = None
    suffixToIdx = None

    def __init__(self, offset) -> None:
        self.suffixToIdx = {}
        self.suffix = ['ing', 'ize', 'ise', 'yse', 'ate', 'ent', 'ent', 'en', 'ify', 'fy', 'ct', 'fine', 'ive', 'ed']
        super().__init__(6 * len(self.suffix), offset)
        for suffix, idx in zip(self.suffix, range(0, len(self.suffix))):
            self.suffixToIdx[suffix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'VB' and tag != 'VBD' and tag != 'VBG' and tag != 'VBN' and tag != 'VBP' and tag != 'VBZ':
            return np.array([])
        for suffix in self.suffix:
            endswith = current_word.endswith(suffix)
            if tag == 'VB' and endswith:
                return np.array([self.suffixToIdx[suffix]])
            if tag == 'VBD' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + len(self.suffix)
            if tag == 'VBG' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (2 * len(self.suffix))
            if tag == 'VBN' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (3 * len(self.suffix))
            if tag == 'VBP' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (4 * len(self.suffix))
            if tag == 'VBZ' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (5 * len(self.suffix))
        return np.array([])

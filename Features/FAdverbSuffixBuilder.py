import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FAdverbSuffixBuilder(FeatureBuilderBase):
    suffix = None
    suffixToIdx = None

    def __init__(self, offset) -> None:
        self.suffixToIdx = {}
        self.suffix = ['ly', 'ily', 'ely', 'ingly']
        super().__init__(3 * len(self.suffix), offset)
        for suffix, idx in zip(self.suffix, range(0, len(self.suffix))):
            self.suffixToIdx[suffix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'RB' and tag != 'RBR' and tag != 'RBS':
            return np.array([])
        for suffix in self.suffix:
            endswith = current_word.endswith(suffix)
            if tag == 'RB' and endswith:
                return np.array([self.suffixToIdx[suffix]])
            if tag == 'RBR' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + len(self.suffix)
            if tag == 'RBS' and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (2 * len(self.suffix))
        return np.array([])

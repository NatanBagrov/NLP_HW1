import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FAdverbSuffixBuilder(FeatureBuilderBase):
    suffix = None
    suffixToIdx = {}

    def __init__(self, offset) -> None:
        self.suffix = ['ly', 'ily', 'ely', 'ingly']
        super().__init__(len(self.suffix), offset)
        for suffix, idx in zip(self.suffix, range(0, self.size)):
            self.suffixToIdx[suffix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        res = []
        for suffix in self.suffix:
            if tag.startswith('RB') and current_word.endswith(suffix):
                res = res + [self.suffixToIdx[suffix]]
        return np.array([res])

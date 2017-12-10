import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FNounPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = {}

    def __init__(self, offset) -> None:
        self.prefix = ['non', 'pre']
        super().__init__(4 * len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, self.size)):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        res = []
        for prefix in self.prefix:
            if tag == 'NN' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]]
            if tag == 'NNS' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]] + len(self.prefix)
            if tag == 'NNP' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]] + 2 * len(self.prefix)
            if tag == 'NNPS' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]] + 3 * len(self.prefix)
        return np.array([res])

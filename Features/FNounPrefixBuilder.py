import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FNounPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = None

    def __init__(self, offset) -> None:
        self.prefixToIdx = {}
        self.prefix = ['non', 'pre']
        super().__init__(4 * len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, len(self.prefix))):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'NN' and tag != 'NNS' and tag != 'NNP' and tag != 'NNPS':
            return np.array([])
        for prefix in self.prefix:
            if tag == 'NN' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]])
            if tag == 'NNS' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + len(self.prefix)
            if tag == 'NNP' and current_word.startswith(prefix):
                    return np.array([self.prefixToIdx[prefix]]) + (2 * len(self.prefix))
            if tag == 'NNPS' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + (3 * len(self.prefix))
        return np.array([])

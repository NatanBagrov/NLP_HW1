import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FVerbPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = None

    def __init__(self, offset) -> None:
        self.prefixToIdx = {}
        self.prefix = ['dis', 'mis', 'ob', 'op', 'pre', 'un', 're']
        super().__init__(6 * len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, len(self.prefix))):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'VB' and tag != 'VBD' and tag != 'VBG' and tag != 'VBN' and tag != 'VBP' and tag != 'VBZ':
            return np.array([])
        for prefix in self.prefix:
            if tag == 'VB' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]])
            if tag == 'VBD' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + len(self.prefix)
            if tag == 'VBG' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + (2 * len(self.prefix))
            if tag == 'VBN' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + (3 * len(self.prefix))
            if tag == 'VBP' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + (4 * len(self.prefix))
            if tag == 'VBZ' and current_word.startswith(prefix):
                return np.array([self.prefixToIdx[prefix]]) + (5 * len(self.prefix))
        return np.array([])

import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FAdjPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = None

    def __init__(self, offset) -> None:
        self.prefixToIdx = {}
        self.prefix = ['anti', 'en', 'il', 'im', 'in', 'ir', 'non', 'pre', 'un', 'ly']
        super().__init__(3 * len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, len(self.prefix))):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'JJ' and tag != 'JJR' and tag != 'JJS':
            return np.array([])
        for prefix in self.prefix:
            startswith = current_word.startswith(prefix)
            if tag == 'JJ' and startswith:
                return np.array([self.prefixToIdx[prefix]])
            if tag == 'JJR' and startswith:
                return np.array([self.prefixToIdx[prefix]]) + len(self.prefix)
            if tag == 'JJS' and startswith:
                return np.array([self.prefixToIdx[prefix]]) + (2 * len(self.prefix))
        return np.array([])

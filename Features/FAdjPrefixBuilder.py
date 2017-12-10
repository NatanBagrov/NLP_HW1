import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FAdjPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = {}

    def __init__(self, offset) -> None:
        self.prefix = ['anti', 'en', 'il', 'im', 'in', 'ir', 'non', 'pre', 'un', 'ly']
        super().__init__(3 * len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, self.size)):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        res = []
        for prefix in self.prefix:
            if tag == 'JJ' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]]
            if tag == 'JJR' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]] + len(self.prefix)
            if tag == 'JJS' and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]] + 2 * len(self.prefix)
        return np.array([res])

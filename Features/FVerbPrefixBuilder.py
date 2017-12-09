import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FVerbPrefixBuilder(FeatureBuilderBase):
    prefix = None
    prefixToIdx = {}

    def __init__(self, offset) -> None:
        self.prefix = ['dis','mis','ob','op','pre','un','re']
        super().__init__(len(self.prefix), offset)
        for prefix, idx in zip(self.prefix, range(0, self.size)):
            self.prefixToIdx[prefix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        res = []
        for prefix in self.prefix:
            if tag.startswith('VB') and current_word.startswith(prefix):
                res = res + [self.prefixToIdx[prefix]]
        return np.array([res])
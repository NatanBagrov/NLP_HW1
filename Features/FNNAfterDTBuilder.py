import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FNNAfterDTBuilder(FeatureBuilderBase):
    def __init__(self, offset) -> None:
        super().__init__(4, offset)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        t1 = history.t1
        if t1 != 'DT':
            return np.array([])
        if t1 == 'DT' and tag == 'NN':
            return np.array([0]) + self.offset
        if t1 == 'DT' and tag == 'NNS':
            return np.array([0]) + self.offset + 1
        if t1 == 'DT' and tag == 'NNP':
            return np.array([0]) + self.offset + 2
        if t1 == 'DT' and tag == 'NNPS':
            return np.array([0]) + self.offset + 3
        return np.array([])

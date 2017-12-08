import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FNNAfterDTBuilder(FeatureBuilderBase):
    def __init__(self) -> None:
        super().__init__(1)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        t1 = history.t1
        if t1=='DT' and tag[0:2]=='NN':
            return np.array([0])
        return np.array([])
import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class F102Builder(FeatureBuilderBase):
    d = {}

    def __init__(self) -> None:
        super().__init__(1)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag=='NN' and current_word[0:3]=='pre':
            return np.array([0])
        return np.array([])

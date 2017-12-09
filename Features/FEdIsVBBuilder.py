import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FEdIsVbBuilder(FeatureBuilderBase):
    def __init__(self) -> None:
        super().__init__(1)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if current_word[-2:]=='ed' and tag[0:2]=='VB':
            return np.array([0])
        return np.array([])
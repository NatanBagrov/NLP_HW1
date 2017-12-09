import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FAbleBuilder(FeatureBuilderBase):


    def __init__(self,offset) -> None:
        super().__init__(1,offset)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag=='JJ' and current_word[-4:]=='able':
            return np.array([0]) + self.offset
        return np.array([])

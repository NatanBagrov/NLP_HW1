import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FCapitalBuilder(FeatureBuilderBase):

    def __init__(self,offset) -> None:
        super().__init__(1,offset)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if not current_word.islower():
            return np.array([0]) + self.offset
        return np.array([])

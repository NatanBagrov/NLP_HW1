import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class F101Builder(FeatureBuilderBase):


    def __init__(self) -> None:
        super().__init__(1)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag=='VBG' and current_word[-3:]=='ing':
            return np.empty(1)
        return np.array([])

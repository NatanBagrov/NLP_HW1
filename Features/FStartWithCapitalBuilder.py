import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FStartWithCapitalBuilder(FeatureBuilderBase):

    def __init__(self, offset) -> None:
        super().__init__(4, offset)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag == 'NN' and current_word[0].isupper():
            return np.array([0]) + self.offset
        if tag == 'NNS' and current_word[0].isupper():
            return np.array([0]) + self.offset + 1
        if tag == 'NNP' and current_word[0].isupper():
            return np.array([0]) + self.offset + 2
        if tag == 'NNPS' and current_word[0].isupper():
            return np.array([0]) + self.offset + 3
        return np.array([])

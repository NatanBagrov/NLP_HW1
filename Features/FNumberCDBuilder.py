import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FNumberCDBuilder(FeatureBuilderBase):
    digits = ['one', 'two', 'zero', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    def __init__(self,offset) -> None:
        super().__init__(1,offset)

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        if tag != 'CD':
            return np.array([])
        current_word = str(history.sentence[history.idx]).lower()
        if any(i.isdigit() for i in current_word) and tag=='CD':
            return np.array([0]) + self.offset
        if current_word in self.digits and tag=='CD':
            return np.array([0]) + self.offset
        return np.array([])

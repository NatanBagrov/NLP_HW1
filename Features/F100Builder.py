import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class F100Builder(FeatureBuilderBase):
    d={}
    def __init__(self,words) -> None:
        super().__init__(len(words))
        self.d = {}
        for index in range(0,self.size):
            self.d[words[index]]=index

    def getFeatureVector(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        tpl = (current_word, tag)
        if tpl in self.d.keys():
            return np.array([self.d[tpl]])
        return np.array([])

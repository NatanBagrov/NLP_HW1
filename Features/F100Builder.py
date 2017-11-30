import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class F100Builder(FeatureBuilderBase):
    d={}
    def __init__(self,words) -> None:
        super().__init__(len(words))
        for index in range(0,self.size):
            self.d[words[index]]=index

    def getFeatureVector(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        vec=np.zeros(self.size)
        tpl = (current_word, tag)
        if tpl in self.d.keys():
            vec[self.d[tpl]]=1
        return vec


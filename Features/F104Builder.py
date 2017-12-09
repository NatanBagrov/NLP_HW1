import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class F104Builder(FeatureBuilderBase):
    d={}
    size=None

    def __init__(self, tags, offset) -> None:
        super().__init__(len(tags), offset)
        self.d = {}
        for index in range(0,self.size):
            self.d[tags[index]]=index

    def getFeatureVector(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        tpl = (history.t1, tag)
        if tpl in self.d:
            return np.array([self.d[tpl]]) + self.offset
        return np.array([])

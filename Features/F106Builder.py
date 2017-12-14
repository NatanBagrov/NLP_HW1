import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class F106Builder(FeatureBuilderBase):
    d={}
    def __init__(self, uniqueTags, offset) -> None:
        super().__init__(len(uniqueTags), offset)
        self.d = {}
        for index in range(0,self.size):
            self.d[uniqueTags[index]] = index + offset

    def getFeatureVector(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        if tag in self.d:
            return np.array([self.d[tag]])
        return np.array([])

    def getFeatureVectorTrain(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        if tag in self.d:
            return np.array([self.d[tag]])
        return np.array([])
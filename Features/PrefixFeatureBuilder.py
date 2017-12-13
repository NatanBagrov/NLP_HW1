import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class PrefixFeatureBuilder(FeatureBuilderBase):
    d_train = {}
    d_inference = {}
    prefixes = None

    def __init__(self,parser: MyParser, offset) -> None:
        self.d_train = {}
        self.d_inference = {}
        self.prefixes = ['anti', 'en', 'il', 'im', 'in', 'ir', 'non', 'pre', 'un', 'ly', 'dis', 'mis', 'ob', 'op', 're']
        self.prefixes = sorted(set(self.prefixes))
        pref_triplet = parser.getAllTagsForPrefix(self.prefixes)
        pref_len = len(pref_triplet)
        for (w,x,t),i in zip(pref_triplet, range(0, pref_len)):
            if (x,t) not in self.d_inference:
                self.d_inference[(x,t)] = i + offset
            self.d_train[(w,t)] = self.d_inference[(x,t)]

        super().__init__(len(self.d_inference),offset)



    def getFeatureVectorTrain(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        tpl = (current_word, tag)
        if tpl in self.d_train:
            return np.array([self.d_train[tpl]])
        return np.array([])

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        for pref in self.prefixes:
            if current_word.startswith(pref):
                tpl = (pref, tag)
                if tpl in self.d_inference:
                    return np.array([self.d_inference[tpl]])
                return np.array([])
        return np.array([])

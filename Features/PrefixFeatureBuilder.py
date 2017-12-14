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
        self.prefixes = ['a', 'ante', 'anti', 'arch', 'auto', 'bi', 'circum', 'co', 'col', 'com',
                         'con', 'contra','counter', 'de', 'dia', 'dis', 'dys', 'e', 'eco', 'en', 'em', 'equi', 'ex',
                         'extra', 'fore', 'hyper', 'il', 'im', 'in', 'ir', 'inter', 'inrta', 'kilo', 'macro',
                         'mal', 'micro', 'mid', 'mis', 'mono', 'multi', 'neo', 'non', 'out', 'over', 'post', 'pre',
                         'pro', 'pseudo', 're', 'retro', 'semi', 'sub', 'super', 'trans', 'ultra', 'un', 'under', 'well']
        self.prefixes = sorted(set(self.prefixes))
        pref_triplet = parser.getAllTagsForPrefix(self.prefixes)
        self.prefixes = []
        pref_len = len(pref_triplet)
        for (w,x,t),i in zip(pref_triplet, range(0, pref_len)):
            if x not in self.prefixes:
                self.prefixes.append(x)
            if (x,t) not in self.d_inference:
                self.d_inference[(x,t)] = len(self.d_inference) + offset
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

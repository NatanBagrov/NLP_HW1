import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class SuffixFeatureBuilder(FeatureBuilderBase):
    d_train = {}
    d_inference = {}
    suffixes = None

    def __init__(self, parser: MyParser, offset) -> None:
        self.d_train = {}
        self.d_inference = {}
        self.suffixes = ['ful', 'ive', 'ic', 'al', 'able', 'ed', 'ible', 'ing', 'ous', 'ish', 'like', 'some',
                         'worthy', 'ly', 'ily', 'ely', 'ingly', 'ment', 'ness', 'sion', 'tion', 'ty', 'al', 'ance',
                         'hood', 'dom', 'ght', 'ful', 'er', 'age', 'sis', 'ism', 'ity', 'ant', 'ssion', 'ship', 'th',
                         'cess', 'ize', 'ise', 'yse', 'ate', 'ent', 'en', 'ify', 'fy', 'ct', 'fine', 'ive', 'ed']
        self.suffixes = sorted(set(self.suffixes))
        suf_triplet = parser.getAllTagsForSuffix(self.suffixes)
        self.suffixes = []
        suf_len = len(suf_triplet)
        for (w,x,t),i in zip(suf_triplet, range(0, suf_len)):
            if x not in self.suffixes:
                self.suffixes.append(x)
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
        for sef in self.suffixes:
            if current_word.endswith(sef):
                tpl = (sef, tag)
                if tpl in self.d_inference:
                    return np.array([self.d_inference[tpl]])
                return np.array([])
        return np.array([])

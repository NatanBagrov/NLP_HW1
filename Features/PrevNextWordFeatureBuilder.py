import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class PrevNextWordFeatureBuilder(FeatureBuilderBase):
    d_prev={}
    d_next = {}
    size=None

    def __init__(self, prevWordCurrTags, nextWordCurrTags, offset) -> None:

        super().__init__(len(prevWordCurrTags) + len(nextWordCurrTags), offset)
        self.d_prev = {}
        self.d_next = {}
        for index in range(0,len(prevWordCurrTags)):
            self.d_prev[prevWordCurrTags[index]]=index + offset
        for index in range(0,len(nextWordCurrTags)):
            self.d_next[prevWordCurrTags[index]]=index+len(prevWordCurrTags) + offset

    def getFeatureVector(self,history,tag): #history=(t-2,t-1,list of words in sentence, index)
        #Prev word with curr tag
        if history.idx == 0:
            prev_w = '*'
        else:
            prev_w = history.sentence[history.idx - 1]
        tpl = (prev_w,tag)
        res = []
        if tpl in self.d_prev:
            res = res + [self.d_prev[tpl]]

        #Next word with curr tag
        if history.idx == (len(history.sentence) - 1):
            next_w = "SEN-END"
        else:
            next_w = history.sentence[history.idx + 1]
        tpl = (next_w, tag)
        if tpl in self.d_next:
            res = res + [self.d_next[tpl]]
        return np.array(res)


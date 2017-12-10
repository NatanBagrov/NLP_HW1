import numpy as np
from Features.FeatureBuilderBase import FeatureBuilderBase


class FAdjSuffixBuilder(FeatureBuilderBase):
    suffix = None
    suffixToIdx = None

    def __init__(self, offset) -> None:
        self.suffixToIdx = {}
        self.suffix = ['ful', 'ive', 'ic', 'al', 'able', 'ed', 'ible', 'ing', 'ous', 'ish', 'ly', 'like', 'some',
                       'worthy']
        super().__init__(3 * len(self.suffix), offset)
        for suffix, idx in zip(self.suffix, range(0, len(self.suffix))):
            self.suffixToIdx[suffix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        res = []
        for suffix in self.suffix:
            if tag == 'JJ' and current_word.endswith(suffix):
                res = res + [self.suffixToIdx[suffix]]
            if tag == 'JJR' and current_word.endswith(suffix):
                res = res + [self.suffixToIdx[suffix] + len(self.suffix)]
            if tag == 'JJS' and current_word.endswith(suffix):
                res = res + [self.suffixToIdx[suffix] + 2 * len(self.suffix)]
        return np.array(res)

import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase


class FNounSuffixBuilder(FeatureBuilderBase):
    suffix = None
    suffixToIdx = None

    def __init__(self, offset) -> None:
        self.suffixToIdx = {}
        self.suffix = ['ment', 'ness', 'sion', 'tion', 'ty', 'al', 'ance', 'hood', 'dom', 'ght', 'ful', 'er', 'age',
                       'sis', 'ism', 'ity', 'ant', 'ssion', 'ship', 'th', 'cess']
        super().__init__(4 * len(self.suffix), offset)
        for suffix, idx in zip(self.suffix, range(0, len(self.suffix))):
            self.suffixToIdx[suffix] = idx + offset

    def getFeatureVector(self, history, tag):  # history=(t-2,t-1,list of words in sentence, index)
        current_word = history.sentence[history.idx]
        if tag != 'NN' and tag != 'NNS' and tag != 'NNP' and tag != 'NNPS':
            return np.array([])
        for suffix in self.suffix:
            endswith = current_word.endswith(suffix)
            if tag.startswith('NN') and endswith:
                return np.array([self.suffixToIdx[suffix]])
            if tag.startswith('NNS') and endswith:
                return np.array([self.suffixToIdx[suffix]]) + len(self.suffix)
            if tag.startswith('NNP') and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (2 * len(self.suffix))
            if tag.startswith('NNPS') and endswith:
                return np.array([self.suffixToIdx[suffix]]) + (3 * len(self.suffix))
        return np.array([])

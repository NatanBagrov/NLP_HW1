from Features.F100Builder import F100Builder
from Features.F103Builder import F103Builder
from Features.F104Builder import F104Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser
import numpy as np


class BasicFeatureVectorBuilder(FeatureBuilderBase):
    parser = None
    f100 = None
    f103 = None
    f104 = None

    def __init__(self, parser: MyParser) -> None:
        self.parser = parser
        self.f100 = F100Builder(parser.getWordsWithTag())
        self.f103 = F103Builder(parser.getAllThreeTagsCombinations())
        self.f104 = F104Builder(parser.getAllPairTagsCombinations())
        super().__init__(self.f100.size + self.f103.size + self.f104.size)

    def getFeatureVector(self, history, tag):
        vec100 = self.f100.getFeatureVector(history, tag)
        vec103 = self.f103.getFeatureVector(history, tag) + self.f100.size
        vec104 = self.f104.getFeatureVector(history, tag) + self.f100.size + self.f103.size
        return np.concatenate((vec100, vec103, vec104)).astype(int)

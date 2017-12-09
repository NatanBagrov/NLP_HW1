from Features.F100Builder import F100Builder
from Features.F103Builder import F103Builder
from Features.F104Builder import F104Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser
import numpy as np


class BasicFeatureVectorBuilder(FeatureBuilderBase):
    parser = None
    offset = None
    f100 = None
    f103 = None
    f104 = None

    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        self.f100 = F100Builder(parser.getWordsWithTag(),0)
        self.f103 = F103Builder(parser.getAllThreeTagsCombinations(),self.f100.size)
        self.f104 = F104Builder(parser.getAllPairTagsCombinations(),self.f100.size + self.f103.size)
        super().__init__(self.f100.size + self.f103.size + self.f104.size, offset)

    def getFeatureVector(self, history, tag):
        vec100 = self.f100.getFeatureVector(history, tag)
        vec103 = self.f103.getFeatureVector(history, tag)
        vec104 = self.f104.getFeatureVector(history, tag)
        return np.concatenate((vec100, vec103, vec104)).astype(int)

    def getFeatureVectors(self, history, tag):
        vec100 = self.f100.getFeatureVectors(history, tag)
        vec103 = self.f103.getFeatureVectors(history, tag)
        vec104 = self.f104.getFeatureVectors(history, tag)
        res = [np.concatenate((v1, v2, v3)).astype(int) for v1,v2,v3 in zip(vec100,vec103,vec104)]
        return res

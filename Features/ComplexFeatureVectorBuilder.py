import numpy as np

from Features.CapsFeatureBuilder import CapsFeatureBuilder
from Features.DigitNumberFeatureBuilder import DigitNumberFeatureBuilder
from Features.DigitWordFeatureBuilder import DigitWordFeatureBuilder
from Features.PrefixFeatureBuilder import PrefixFeatureBuilder
from Features.F100Builder import F100Builder
from Features.F103Builder import F103Builder
from Features.F104Builder import F104Builder
from Features.F106Builder import F106Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from Features.SuffixFeatureBuilder import SuffixFeatureBuilder
from MyParser import MyParser


class ComplexFeatureVectorBuilder(FeatureBuilderBase):
    parser = None
    f100 = None
    f103 = None
    f104 = None
    fSuf = None
    fPref = None
    fDigNum = None
    fLetNum = None
    fCaps = None
    isTraining = None

    def __init__(self, parser: MyParser, train_parser: MyParser, isTraining) -> None:
        self.parser = parser
        self.isTraining = isTraining
        vecSize = 0

        self.f100 = F100Builder(parser.getWordsWithTag(), vecSize)
        vecSize = self.f100.size
        print("F100 size", self.f100.size)

        self.f103 = F103Builder(parser.getAllThreeTagsCombinations(), vecSize)
        vecSize = vecSize + self.f103.size
        print("F103 size", self.f103.size)

        self.f104 = F104Builder(parser.getAllPairTagsCombinations(), vecSize)
        vecSize = vecSize + self.f104.size
        print("F104 size", self.f104.size)

        self.f106 = F106Builder(parser.getUniqueTags(), vecSize)
        vecSize = vecSize + self.f106.size
        print("F106 size", self.f106.size)

        self.fSuf = SuffixFeatureBuilder(vecSize, train_parser)
        vecSize = vecSize + self.fSuf.size
        print("Suffix size", self.fSuf.size)

        self.fPref = PrefixFeatureBuilder(vecSize, train_parser)
        vecSize = vecSize + self.fPref.size
        print("Prefix size", self.fPref.size)

        self.fDigNum = DigitNumberFeatureBuilder(vecSize, train_parser)
        vecSize = vecSize + self.fDigNum.size
        print("DigitNum size", self.fDigNum.size)

        self.fLetNum = DigitWordFeatureBuilder(vecSize, train_parser)
        vecSize = vecSize + self.fLetNum.size
        print("DigitLetter size", self.fLetNum.size)

        self.fCaps = CapsFeatureBuilder(vecSize, train_parser)
        vecSize = vecSize + self.fCaps.size
        print("Caps size", self.fCaps.size)

        super().__init__(vecSize, 0)

    def getFeatureVector(self, history, tag):
        vec100 = self.f100.getFeatureVector(history, tag)
        vec103 = self.f103.getFeatureVector(history, tag)
        vec104 = self.f104.getFeatureVector(history, tag)
        vec106 = self.f106.getFeatureVector(history, tag)

        if self.isTraining:
            vecSuf = self.fSuf.getFeatureVectorTrain(history,tag)
            vecPref = self.fPref.getFeatureVectorTrain(history,tag)
            vecDigNum = self.fDigNum.getFeatureVectorTrain(history,tag)
            vecDigLet = self.fLetNum.getFeatureVectorTrain(history,tag)
            vecCaps = self.fCaps.getFeatureVectorTrain(history,tag)
        else:
            vecSuf = self.fSuf.getFeatureVector(history,tag)
            vecPref = self.fPref.getFeatureVector(history,tag)
            vecDigNum = self.fDigNum.getFeatureVector(history,tag)
            vecDigLet = self.fLetNum.getFeatureVector(history,tag)
            vecCaps = self.fCaps.getFeatureVector(history,tag)

        return np.concatenate(
            (vec100, vec103, vec104, vec106, vecSuf, vecPref, vecDigNum, vecDigLet, vecCaps)).astype(int)

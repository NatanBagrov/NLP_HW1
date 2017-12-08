from Features.F100Builder import F100Builder
from Features.F101Builder import F101Builder
from Features.F102Builder import F102Builder
from Features.F103Builder import F103Builder
from Features.F105Builder import F105Builder
from Features.F104Builder import F104Builder
from Features.FAbleBuilder import FAbleBuilder
from Features.FCapitalBuilder import FCapitalBuilder
from Features.FLyJJBuilder import FLyJJBuilder
from Features.FLyRBBuilder import FLyRBBuilder
from Features.FNumberBuilder import FNumberBuilder
from Features.FNumberCDBuilder import FNumberCDBuilder
from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser
import numpy as np


class ComplexFeatureVectorBuilder(FeatureBuilderBase):
    parser = None
    f100 = None
    f101 = None
    f102 = None
    f103 = None
    f104 = None
    f105 = None
    fCapital = None
    fNumber = None
    fNumberCD = None
    fAble = None
    fLyJJ = None
    fLyRB = None

    def __init__(self, parser: MyParser) -> None:
        self.parser = parser
        self.f100 = F100Builder(parser.getWordsWithTag())
        self.f101 = F101Builder()
        self.f102 = F102Builder()
        self.f103 = F103Builder(parser.getAllThreeTagsCombinations())
        self.f104 = F104Builder(parser.getAllPairTagsCombinations())
        self.f105 = F105Builder(parser.getUniqueTags())
        self.fCapital = FCapitalBuilder()
        self.fNumber = FNumberBuilder()
        self.fNumberCD = FNumberCDBuilder()
        self.fAble = FAbleBuilder()
        self.fLyJJ = FLyJJBuilder()
        self.fLyRB = FLyRBBuilder()
        super().__init__(self.f100.size + self.f101.size + self.f102.size + self.f103.size + self.f104.size
                         + self.f105.size + self.fCapital.size + self.fNumber.size + self.fNumberCD.size + self.fAble.size
                         + self.fLyJJ.size + self.fLyRB.size)

    def getFeatureVector(self, history, tag):
        vec100 = self.f100.getFeatureVector(history, tag)
        vecSize = self.f100.size
        vec101 = self.f101.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.f101.size
        vec102 = self.f102.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.f102.size
        vec103 = self.f103.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.f103.size
        vec104 = self.f104.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.f104.size
        vec105 = self.f105.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.f105.size
        vecCapital = self.fCapital.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.fCapital.size
        vecNumber = self.fNumber.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.fNumber.size
        vecNumberCD = self.fNumberCD.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.fNumberCD.size
        vecAble = self.fAble.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.fAble.size
        vecLyJJ = self.fLyJJ.getFeatureVector(history, tag) + vecSize
        vecSize = vecSize + self.fLyJJ.size
        vecLyRB = self.fLyRB.getFeatureVector(history, tag) + vecSize
        return np.concatenate(
            (vec100, vec101, vec102, vec103, vec104, vec105, vecCapital, vecNumber, vecNumberCD, vecAble, vecLyJJ,
             vecLyRB)).astype(int)

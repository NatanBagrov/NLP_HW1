from Features.F100Builder import F100Builder
from Features.F101Builder import F101Builder
from Features.F102Builder import F102Builder
from Features.F103Builder import F103Builder
from Features.F105Builder import F105Builder
from Features.F104Builder import F104Builder
from Features.FAdjPrefixBuilder import FAdjPrefixBuilder
from Features.FAdjSuffixBuilder import FAdjSuffixBuilder
from Features.FAdverbSuffixBuilder import FAdverbSuffixBuilder
from Features.FCapitalBuilder import FCapitalBuilder
from Features.FNNAfterDTBuilder import FNNAfterDTBuilder
from Features.FNounPrefixBuilder import FNounPrefixBuilder
from Features.FNounSuffixBuilder import FNounSuffixBuilder
from Features.FNumberBuilder import FNumberBuilder
from Features.FNumberCDBuilder import FNumberCDBuilder
from Features.FStartWithCapitalBuilder import FStartWithCapitalBuilder
from Features.FVerbPrefixBuilder import FVerbPrefixBuilder
from Features.FVerbSuffixBuilder import FVerbSuffixBuilder
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
    fAdjPrefix = None
    fAdjSuffix = None
    fAdverbSuffix = None
    fNNAfterDT = None
    fNounPrefix = None
    fNounSuffix = None
    fCapitalFirst = None
    fVerbPrefix = None
    fVerbSuffix = None

    def __init__(self, parser: MyParser) -> None:
        self.parser = parser
        vecSize = 0
        self.f100 = F100Builder(parser.getWordsWithTag(), vecSize)
        vecSize = self.f100.size
        self.f101 = F101Builder(vecSize)
        vecSize = vecSize + self.f101.size
        self.f102 = F102Builder(vecSize)
        vecSize = vecSize + self.f102.size
        self.f103 = F103Builder(parser.getAllThreeTagsCombinations(), vecSize)
        vecSize = vecSize + self.f103.size
        self.f104 = F104Builder(parser.getAllPairTagsCombinations(), vecSize)
        vecSize = vecSize + self.f104.size
        self.f105 = F105Builder(parser.getUniqueTags(), vecSize)
        vecSize = vecSize + self.f105.size
        self.fCapital = FCapitalBuilder(vecSize)
        vecSize = vecSize + self.fCapital.size
        self.fNumber = FNumberBuilder(vecSize)
        vecSize = vecSize + self.fNumber.size
        self.fNumberCD = FNumberCDBuilder(vecSize)
        vecSize = vecSize + self.fNumberCD.size
        self.fAdjPrefix = FAdjPrefixBuilder(vecSize)
        vecSize = vecSize + self.fAdjPrefix.size
        self.fAdjSuffix = FAdjSuffixBuilder(vecSize)
        vecSize = vecSize + self.fAdjSuffix.size
        self.fAdverbSuffix = FAdverbSuffixBuilder(vecSize)
        vecSize = vecSize + self.fAdverbSuffix.size
        self.fNNAfterDT = FNNAfterDTBuilder(vecSize)
        vecSize = vecSize + self.fNNAfterDT.size
        self.fNounPrefix = FNounPrefixBuilder(vecSize)
        vecSize = vecSize + self.fNounPrefix.size
        self.fNounSuffix = FNounSuffixBuilder(vecSize)
        vecSize = vecSize + self.fNounSuffix.size
        self.fCapitalFirst = FStartWithCapitalBuilder(vecSize)
        vecSize = vecSize + self.fCapitalFirst.size
        self.fVerbPrefix = FVerbPrefixBuilder(vecSize)
        vecSize = vecSize + self.fVerbPrefix.size
        self.fVerbSuffix = FVerbSuffixBuilder(vecSize)
        vecSize = vecSize + self.fVerbSuffix.size
        super().__init__(vecSize, 0)

    def getFeatureVector(self, history, tag):
        vec100 = self.f100.getFeatureVector(history, tag)
        vec101 = self.f101.getFeatureVector(history, tag)
        vec102 = self.f102.getFeatureVector(history, tag)
        vec103 = self.f103.getFeatureVector(history, tag)
        vec104 = self.f104.getFeatureVector(history, tag)
        vec105 = self.f105.getFeatureVector(history, tag)
        vecCapital = self.fCapital.getFeatureVector(history, tag)
        vecNumber = self.fNumber.getFeatureVector(history, tag)
        vecNumberCD = self.fNumberCD.getFeatureVector(history, tag)
        vecAdjPrefix = self.fAdjPrefix.getFeatureVector(history, tag)
        vecAdjSuffix = self.fAdjSuffix.getFeatureVector(history, tag)
        vecAdverbSuffix = self.fAdverbSuffix.getFeatureVector(history, tag)
        vecNNAfterDT = self.fNNAfterDT.getFeatureVector(history, tag)
        vecNounPrefix = self.fNounPrefix.getFeatureVector(history, tag)
        vecNounSuffix = self.fNounSuffix.getFeatureVector(history, tag)
        vecCapitalFirst = self.fCapitalFirst.getFeatureVector(history, tag)
        vecVerbPrefix = self.fVerbPrefix.getFeatureVector(history, tag)
        vecVerbSuffix = self.fVerbSuffix.getFeatureVector(history, tag)
        return np.concatenate(
            (vec100, vec101, vec102, vec103, vec104, vec105, vecCapital, vecNumber, vecNumberCD
             , vecAdjPrefix, vecAdjSuffix, vecAdverbSuffix, vecNNAfterDT, vecNounPrefix, vecNounSuffix,
             vecCapitalFirst, vecVerbPrefix, vecVerbSuffix)).astype(int)

import abc
from abc import ABCMeta

import numpy as np


class FeatureBuilderBase(metaclass=ABCMeta):
    size = None

    def __init__(self, size, offset) -> None:
        super().__init__()
        self.size = size
        self.offset = offset

    @abc.abstractmethod
    def getFeatureVector(self, history, tag):
        raise NotImplemented

    # def getFeatureVectors(self, history, tags):
    #     return [self.getFeatureVector(history, t) for t in tags]

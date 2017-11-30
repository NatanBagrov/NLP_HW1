import abc
from abc import ABCMeta


class FeatureBuilderBase(metaclass=ABCMeta):
    size = None

    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    @abc.abstractmethod
    def getFeatureVector(self, history, tag):
        raise NotImplemented

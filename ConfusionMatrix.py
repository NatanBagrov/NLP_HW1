import numpy as np


class ConfusionMatrix():
    tags = None
    size = None
    tagsToIdx = {}
    idxToTags = {}

    def __init__(self, tags) -> None:
        super().__init__()
        self.tags = tags
        self.size = len(tags)
        for tag, idx in zip(self.tags, range(0, len(tags))):
            self.tagsToIdx[tag] = idx
            self.idxToTags[idx] = tag

    def buildMatrix(self, expected, actual):
        mat = np.zeros((self.size, self.size))
        for e, a in zip(expected, actual):
            e, a = e.strip().split(), a.strip().split()
            for t1, t2 in zip(e, a):
                row = self.tagsToIdx[t1]
                col = self.tagsToIdx[t2]
                idx = row * self.size + col
                np.put(mat, [idx], np.take(mat, idx) + 1)
        return mat

    def calcAccuracy(self, matrix):
        sums = np.sum(matrix, axis=1, dtype=float)
        diagnol = np.diagonal(matrix)
        return diagnol / sums

    def calculateMatrixForLowestNTags(self, expected, actual, n):
        matrix = self.buildMatrix(expected, actual)
        indices = self.calcAccuracy(matrix).argsort()[:n]
        res = [self.idxToTags[idx] for idx in indices]
        return matrix[indices], res

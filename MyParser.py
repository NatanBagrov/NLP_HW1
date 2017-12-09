from itertools import chain


class MyParser:
    fileName = None
    lines = None
    splitted = None

    def __init__(self, fileName):
        super().__init__()
        self.fileName = fileName
        self.lines = [line.rstrip('\n') for line in open(self.fileName)]
        self.lines = [line.split() for line in self.lines]
        self.splitted = []
        for line in self.lines:
            self.splitted.append([self.createTuple(x) for x in line])

    def createTuple(self, x):
        return (x.split('_')[0], x.split('_')[1])

    def getWordsWithTag(self):
        l = list(set(list(chain.from_iterable(self.splitted))))
        l.sort()
        return l

    def getAllThreeTagsCombinations(self):
        tags = []
        for line in self.splitted:
            for (_, t1), (_, t2), (_, t3) in zip(line[:], line[1:], line[2:]):
                tags.append((t1, t2, t3))
            tags.append(("*", "*", line[0][1]))
            if len(line) > 1:
                tags.append(("*", line[0][1], line[1][1]))
        l = list(set(tags))
        l.sort()
        return l

    def getAllPairTagsCombinations(self):
        tags = []
        for line in self.splitted:
            for (_, t1), (_, t2) in zip(line[:], line[1:]):
                tags.append((t1, t2))
            tags.append(("*", line[0][1]))
        l = list(set(tags))
        l.sort()
        return l

    def getUniqueTags(self):
        l = self.getWordsWithTag()
        tags = [w[1] for w in l]
        tags = list(set(tags))
        tags.sort()
        return tags

    def getSeenWordsToTagsDict(self):
        words_with_tag = self.getWordsWithTag()
        d = dict()
        for w,t in words_with_tag:
            if w not in d:
                d[w] = [t]
                continue
            if t not in d[w]:
                d[w] = d[w] + [t]
        return d

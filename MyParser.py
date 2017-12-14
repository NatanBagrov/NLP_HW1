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

    def getAllPrevWordTagCombinations(self):
        tags = []
        for line in self.splitted:
            for (w_prev, _), (_, t_curr) in zip(line[:], line[1:]):
                tags.append((w_prev, t_curr))
            tags.append(("*", line[0][1]))
        l = list(set(tags))
        l.sort()
        return l

    def getAllNextWordTagCombinations(self):
        tags = []
        for line in self.splitted:
            for (_, t_curr), (w_next, _) in zip(line[:], line[1:]):
                tags.append((w_next, t_curr))
            tags.append(("SEN-END", line[-1][1]))
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

    def getAllTagsForPrefix(self, prefixes):
        l = self.getWordsWithTag()
        pref = []
        for w,t in l:
            pref = pref + [(w,x,t) for x in prefixes if w.startswith(x)]
        pref = list(set(pref))
        pref.sort()
        return pref

    def getAllTagsForSuffix(self, suffixes):
        l = self.getWordsWithTag()
        suf = []
        for w, t in l:
            suf = suf + [(w, x, t) for x in suffixes if w.endswith(x)]
        suf = list(set(suf))
        suf.sort()
        return suf

    def getAllTagsForLettersNumbers(self, digits):
        res = []
        l = self.getWordsWithTag()
        for w,t in l:
            res = res + [(w,x,t) for x in digits if str(w).startswith(x)]
        return res


    def getAllTagsForCaps(self):
        l = self.getWordsWithTag()
        res = []
        for w, t in l:
            if not str(w).islower():
                res = res + [(w,t)]
        return res


    def getAllTagsForDigitLetters(self):
        res = []
        l = self.getWordsWithTag()
        for w, t in l:
            if any(i.isdigit() for i in str(w)):
                res = res + [(w, t)]
        return res

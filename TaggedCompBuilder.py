from itertools import chain


class TaggedCompBuilder:
    fileName = None
    lines = None
    splitted = None
    parser = None

    def __init__(self, taggedComp, compNonTagged):
        super().__init__()
        myTagged = open('../comp748.wtag', 'w')
        self.fileName = taggedComp
        self.lines = [line.rstrip('\n') for line in open(self.fileName)]
        compSentences = [line.rstrip('\n') for line in open(compNonTagged)]
        count = 0
        for compSentence in compSentences:
            if compSentence in self.lines:
                idx = self.lines.index(compSentence)
                t = self.lines[idx+1]
                for word,tag in zip(compSentence.split(), t.split()):
                    wt = word + '_' + tag + ' '
                    myTagged.write(wt)
                myTagged.write('\n')




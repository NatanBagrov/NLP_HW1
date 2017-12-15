from itertools import chain


def buildDummy(compNonTagged, compDummyTagged):
    myTagged = open(compDummyTagged, 'w')
    compSentences = [line.rstrip('\n') for line in open(compNonTagged)]
    for compSentence in compSentences:
        for word in compSentence.split():
            wt = word + '_X '
            myTagged.write(wt)
        myTagged.write('\n')


def buildCompResult(origFile, tagsFile, outputFile):
    outputFile = open(outputFile, 'w')
    sentences = [line.rstrip('\n') for line in open(origFile)]
    allTags = [line.rstrip('\n') for line in open(tagsFile)]
    for sentence, tags in zip(sentences, allTags):
        for word, tag in zip(sentence.split(), tags.split()):
            outputFile.write(word + "_" + tag + " ")
        outputFile.write("\n")


def sanity(comp748, tagsFile):
    sentences = [line.rstrip('\n') for line in open(comp748)]
    allTags = [line.rstrip('\n') for line in open(tagsFile)]
    count = 0
    for sentence in sentences:
            for tagged in allTags:
                if tagged == sentence:
                    count += 1

    print(count)

if __name__ == '__main__':
    # buildDummy('../comp.words', 'comp_dummy.wtag')
    #buildCompResult('../comp.words', 'basicTagsResults.txt', 'comp_m1_301386900.wtag')
    sanity('../comp748.wtag','comp_m1_301386900.wtag')
    # buildCompResult('../comp.words','complexTagsResults.txt','comp_m2_203995121.wtag')

from myParser import MyParser

p = MyParser("../train.wtag")
words = p.getWordsWithTag()
print(words)
tag3 = p.getAllThreeTagsCombinations()
tag2 = p.getAllPairTagsCombinations()
print(tag3)
print(tag2)

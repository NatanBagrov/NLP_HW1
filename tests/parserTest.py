from MyParser import MyParser

p = MyParser("../train.wtag")
words = p.getWordsWithTag()
tag3 = p.getAllThreeTagsCombinations()
tag2 = p.getAllPairTagsCombinations()
tag = p.getUniqueTags()
# print(tag3)
# print(tag2)
print(tag)

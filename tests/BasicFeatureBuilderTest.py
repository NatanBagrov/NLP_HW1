from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from History import History
from myParser import MyParser


def basicTest():
    parser = MyParser('small.wtag')
    basic = BasicFeatureVectorBuilder(parser)

    history1 = History("t3", "t8",["w2","w2","w5","w3","w13","w31","w33"],2)
    vec1 = basic.getFeatureVector(history1,"t50")
    assert vec1.size == 3
    print (vec1)

    history2 = History("t4", "t8", ["w2", "w2", "w5", "w3", "w13", "w31", "w33"], 2)
    vec2 = basic.getFeatureVector(history2, "t50")
    assert vec2.size == 2
    print(vec2)

    history3 = History("t4", "t8", ["w2", "w2", "w4", "w3", "w13", "w31", "w33"], 2)
    vec3 = basic.getFeatureVector(history3, "t50")
    assert vec3.size == 1
    print(vec3)

    vec4 = basic.getFeatureVector(history3, "noTag")
    assert vec4.size == 0


def trainwtagTest():
    parser = MyParser('../train.wtag')
    basic = BasicFeatureVectorBuilder(parser)
    splitted = parser.splitted[2829]
    sentence = [l[0] for l in splitted]

    history1 = History("IN", "DT", sentence, 11)
    vec1 = basic.getFeatureVector(history1, "NN")
    assert vec1.size == 3
    print(vec1)

    history2 = History("NoTag", "DT", sentence, 11)
    vec2 = basic.getFeatureVector(history2, "NN")
    assert vec2.size == 2
    print(vec2)

    history3 = History("NoTag", "IN", sentence, 11)
    vec3 = basic.getFeatureVector(history3, "DT")
    assert vec3.size == 1
    print(vec3)

    vec4 = basic.getFeatureVector(history3, "noTag")
    assert vec4.size == 0
    print(vec4)


if __name__=='__main__':
    basicTest()
    trainwtagTest()

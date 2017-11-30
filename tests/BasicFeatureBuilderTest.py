from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from History import History
from myParser import MyParser


def basicTest():
    parser = MyParser('small.wtag')
    basic = BasicFeatureVectorBuilder(parser)
    history1 = History("t3","t8",["w2","w2","w5","w3","w13","w31","w33"],2)

    vec1 = basic.getFeatureVector(history1,"t50")
    assert vec1.size == 7+7+7
    assert vec1.sum() == 3
    history2 = History("t4", "t8", ["w2", "w2", "w5", "w3", "w13", "w31", "w33"], 2)
    vec2 = basic.getFeatureVector(history2, "t50")

    assert vec2.sum() == 2
    history3 = History("t4", "t8", ["w2", "w2", "w4", "w3", "w13", "w31", "w33"], 2)
    vec3 = basic.getFeatureVector(history3, "t50")
    assert vec3.sum() == 1

    vec4 = basic.getFeatureVector(history3, "noTag")
    assert vec4.sum() == 0

if __name__=='__main__':
    basicTest()

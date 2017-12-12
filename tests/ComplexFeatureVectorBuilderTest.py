from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from MyParser import MyParser


def feature_num_print():
    parser = MyParser('../train.wtag')
    ComplexFeatureVectorBuilder(parser,parser,True)



if __name__=='__main__':
    feature_num_print()

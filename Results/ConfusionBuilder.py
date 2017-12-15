from ConfusionMatrix import ConfusionMatrix
from MyParser import MyParser


def basicConfusion():
    mp = MyParser("../train.wtag")
    tags = mp.getUniqueTags()
    cm = ConfusionMatrix(tags)
    expected = open('testwtag_expected_finish_basic_opt_v_lambda_0_007.txt')
    actual = open('testwtag_results_finish_basic_opt_v_lambda_0_007.txt')
    mat, res = cm.calculateMatrixForLowestNTags(expected, actual, 10)
    expected.close()
    actual.close()
    output = open('basicConfusionMatrix_141217.txt', 'a')
    for tag in tags:
        output.write(" {}".format(tag))
    output.write('\n')
    for tag, idx in zip(res, range(0, len(res))):
        output.write("{} ".format(tag))
        for j in range(0, mat[idx].size):
            output.write("{} ".format(mat[idx][j]))
        output.write('\n')


def complexConfusion():
    mp = MyParser("../train.wtag")
    tags = mp.getUniqueTags()
    cm = ConfusionMatrix(tags)
    expected = open('testwtag_expected_finish_complex_opt_v_lambda_0_007.txt')
    actual = open('testwtag_results_finish_complex_opt_v_lambda_0_007.txt')
    mat, res = cm.calculateMatrixForLowestNTags(expected, actual, 10)
    expected.close()
    actual.close()
    output = open('complexConfusionMatrix_151217.txt', 'a')
    for tag in tags:
        output.write(" {}".format(tag))
    output.write('\n')
    for tag, idx in zip(res, range(0, len(res))):
        output.write("{} ".format(tag))
        for j in range(0, mat[idx].size):
            output.write("{} ".format(mat[idx][j]))
        output.write('\n')


if __name__ == '__main__':
    basicConfusion()
    complexConfusion()

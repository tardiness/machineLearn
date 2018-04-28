from numpy import *


def loadDataset():
    dataMat = []
    labelMat = []
    fr = open("data/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    #  求转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        #  不断的换 weights 以  实现 sigmoid的值和labelMat的值差距最小(labelMat的值在0,1 之间)
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights


def mytest():
    dataMat, labelMat = loadDataset()
    gradAscent(dataMat, labelMat)


if __name__ == "__main__":
    mytest()

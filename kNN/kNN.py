from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
import operator


def create_data_set():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX,dataSet,labels,k):
    # 欧式距离公式
    dataSetSize = dataSet.shape[0]  #shape[0]  列的长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #相减
    sqDiffMat = diffMat ** 2                       #平方
    sqDistances = sqDiffMat.sum(axis=1)            #相加
    distances = sqDistances**0.5                   #取根号
    sortedDistindicies = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voterlabel = labels[sortedDistindicies[i]]
        classCount[voterlabel] = classCount.get(voterlabel,0)+1
    #排序
    sortedClasscount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClasscount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    #  newValue = {oldValue - min)/(max-min) 归一化特征值
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValue,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))    # 不是矩阵除法   矩阵除法: linalg.solve(matA,matB)
    return normDataSet,ranges,minValue


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix("data/datingTestSet2.txt")
    normMat,ranges,minValue = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCOunt = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i],normMat[numTestVecs:m, :],datingLabels[numTestVecs:m], 3)
        print('the classifier came back with:%d , the real answer is : %d' % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCOunt += 1.0
    print('the total error rate is %f' % (errorCOunt/float(numTestVecs)))


def showResult():
    datingDataMat,datingLabels = file2matrix("data/datingTestSet2.txt")
    # normMat, ranges, labels = autoNorm(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),150.0*array(datingLabels))
    plt.show()


def classifyPerson():
    resultList = ['not at all','in small does','in large dose']
    ffMiles = float(input('frequent flier miles earned per year?'))
    percenTats = float(input('percentage of time spent playing video game?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix("data/datingTestSet2.txt")
    normat,ranges,minValue = autoNorm(datingDataMat)
    inArr = array([ffMiles,percenTats,iceCream])
    # classifierResult = classify0(inArr,datingDataMat,datingLabels,3)
    classifierResult = classify0((inArr-minValue)/ranges,normat,datingLabels,3)
    print('you will probably like this person :',resultList[classifierResult - 1])


def img2vector(filename):
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector


def handWritingCLassTest():
    fileNames = []
    hwLaels = []
    trainingFileList = os.listdir('data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileName.split('_')[0])
        hwLaels.append(classNumber)
        trainingMat[i,:] = img2vector('data/trainingDigits/%s' % fileName)
    testFileList = os.listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileName)
        classifyResult = classify0(vectorUnderTest,trainingMat,hwLaels,3)
        print('the classify came back with:%d , the real is : %d' % (classifyResult, classNumber))
        if classifyResult != classNumber:
            errorCount += 1.0
            fileNames.append(array([fileName,classifyResult,classNumber]))
    print('total error :%d' % errorCount)
    print('error rate is : %f' % (errorCount/float(mTest)))
    print(fileNames)


if __name__ == '__main__':
    # group,labels = create_data_set()
    # print(classify0([0,0],group,labels,3))
    # showResult()
    # datingClassTest()
    # classifyPerson()

    # 手写数字 识别
    handWritingCLassTest()

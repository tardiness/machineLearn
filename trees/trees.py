from math import log


def calcShannonEnt (dataSet):
    #  香农熵
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        #  E(S) =  -sum(pi * log2 (pi))
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #  取从开始到 第 axis 个
            reducedFeatVec = featVec[:axis]
            #  取从 第axis+1个开始 取到 最后
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    # E(S)
    baseEntropy = calcShannonEnt(dataSet)
    bastInfoGain = 0.0
    bastFeature = -1
    for i in range(numFeature):
        #  取dataSet的 第i列数据
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            #  信息增益 计算公式   Gain(S, A)=E(S)–E(S, A)  E(S,A) = sum((|Sv|/|S|)E(Sv))
            prob = len(subDataSet)/float(len(dataSet))
            #  E(S,A)
            newEntropy += prob * calcShannonEnt(subDataSet)
        #  Gain(S,A)
        infoGain = baseEntropy - newEntropy
        if infoGain > bastInfoGain:
            bastInfoGain = infoGain
            bastFeature = i
    return bastFeature


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    # dataSet[0][-1] = 'maybe'
    retData = splitDataSet(dataSet, 0, 1)
    print(calcShannonEnt(retData))
    print(chooseBestFeatureToSplit(dataSet))

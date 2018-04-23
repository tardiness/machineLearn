from math import log
import operator


def calcShannonEnt (dataSet):
    # ID3 算法 以信息熵的下降速度为选取测试属性的标准，即在每个节点选取还尚未被用来划分的具有最高信息增益的属性作为划分标准，
    # 然后继续这个过程，直到生成的决策树能完美分类训练样例
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


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #  如果classList的值全都相等
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #  便利完了都没找到 唯一的分组 选出出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    #  删掉  是因为已经为树添加了一个节点（就是 labels[bestFeat]）
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), sublabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)


def createDataSet():
    #  yes是鱼类 no 不是鱼类
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    #  no surfacing 付出水面能生存  flippers 是否有脚蹼
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    # dataSet[0][-1] = 'maybe'
    retData = splitDataSet(dataSet, 0, 1)
    print(calcShannonEnt(retData))
    print(chooseBestFeatureToSplit(dataSet))
    myTree = createTree(dataSet, labels)
    print(myTree)


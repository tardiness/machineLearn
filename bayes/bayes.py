from numpy import *


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    #  创建空集合
    vocabSet = set([])
    for document in dataSet:
        #  创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my vocalbulary!" % word)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #  为了避免0 乘
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #  f(x) 和 ln(f(x)) 增减性相同 极值点相同
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
    #  这里的相乘是指对应元素相乘，即先将两个向量中的第1个元素相乘，然后将第2个元素相乘
    p1 = sum(vec2classify * p1Vec) + log(pClass1)
    p0 = sum(vec2classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def tetsNB():
    list0Posts, listClasses = loadDataSet()
    myVocabList = createVocabList(list0Posts)
    trainMat = []
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))


def mytest():
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, postingList[3]))
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print(trainMat)
    p0V, p1v, pAb = trainNB0(trainMat, classVec)
    print(p0V, p1v, pAb)


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randomIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    trainMat = []
    trainClass = []
    #  训练
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0v, p1v, pAb = trainNB0(trainMat, array(trainClass))
    #  测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pAb) != classList[docIndex]:
            errorCount += 1
            print("classify error:", docList[docIndex])
    print("the error rate is:", float(errorCount)/len(testSet))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    soretedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return soretedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    fullText = []
    classList = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randomIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pAb = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVocter = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVocter), p0v, p1v, pAb) != classList[docIndex]:
            print("classify error:", docList[docIndex])
            errorCount += 1
    print("the error rate is:", float(errorCount)/len(testSet))
    return vocabList, p0v, p1v


def getTopWords(ny,sf):
    import operator
    vocabList, p0v, p1v = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0v)):
        if p0v[i] > -2.8:
            topSF.append((vocabList[i], p0v[i]))
        if p1v[i] > -4.3:
            topNY.append((vocabList[i], p1v[i]))
    # pair[1]就是每个元组中的第二个元素
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF*******************************************SF")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY*******************************************NY")
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    # tetsNB()
    # spamTest()
    import feedparser
    # ny = feedparser.parse("http://www.englishbaby.com/lessons/rss")
    sf = feedparser.parse("https://sports.yahoo.com/nba/teams/hou/rss.xml")
    ny = feedparser.parse("http://www.nasa.gov/rss/dyn/image_of_the_day.rss")
    vocabList, pSF, pNY = localWords(ny, sf)
    getTopWords(ny, sf)

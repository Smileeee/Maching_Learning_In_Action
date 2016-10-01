#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算经验熵：对分类进行熵计算，不建立在任何特征值的基础上。
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        #首先计算每个类别的个数：
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
#按照给定的特征以及该特征的某个值来提取数据：
#待划分的数据集、划分数据集的特征、特征的返回值。     
def splitDataSet(dataSet, axis, value):
    #为了不修改原始数据集,创建一个新的列表对象
    retDataSet = []
    for featVec in dataSet:
        #在if语句中,程序将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
#选择最大的熵的特征所在的标号：
#信息增益 = 经验熵 - 经验条件熵
def chooseBestFeatureToSplit(dataSet):
    #特征个数：
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    #首先计算经验熵：
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        #这里面内嵌了一个for循环，是查询某个特征值有多少种可能的值：
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        #去掉列表中重复的元素：
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        #计算该特征值的经验条件熵：
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   
        #信息增益 = 经验熵 - 经验条件熵
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        #选择熵最大的特征值
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

#投票表决代码：返回个数最多的那个分类
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #用operator来辅助排序：
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建决策树：
def createTree(dataSet,labels):
    #取每一项的分类：
    classList = [example[-1] for example in dataSet]
    #如果第一个分类的个数等于列表的长度，那就表示只有一个分类
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    #如果如用完了所有的特征却仍然有多种分类，那就采用投票表决的方法：
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    #如过能到这里，说明决策树还没有建完，那就选择最优的特征，继续进行建树：
    bestFeat = chooseBestFeatureToSplit(dataSet) #返回下标值
    bestFeatLabel = labels[bestFeat] #获取最优的特征
    #字典变量myTree存储了即将要建的子树的所有信息，方便后面制图：
    #字典变量，分号钱表示key，分号后表示value，也是一个字典：可以是子树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #遍历当前特征所有的可能值：
    featValues = [example[bestFeat] for example in dataSet]
    #去掉重复值：
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #这行代码复制了类标签,并将其存储在新列表变量中：
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        #树的第一维是特征，第二维是特征的值：建子树时分割数据集：
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

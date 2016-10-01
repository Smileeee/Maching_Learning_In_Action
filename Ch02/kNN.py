#-*- coding: UTF-8 -*- 
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''

from numpy import *
import operator
from os import listdir

from numpy import *  
import operator  
  
'''
文件说明：在python交互界面输入：
>>> import kNN
>>> group,labels=kNN.createDataSet()
>>> kNN.classify0([0,0], group, labels, 3)

>>> reload(kNN)
>>> datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')

>>> reload(kNN)
>>> normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

>>> reload(kNN)
>>> kNN.datingClassTest()


'''

def createDataSet():  
    #我觉得可以这样理解，每一种方括号都是一个维度（秩）list，这里就是二维数组，最里面括着每一行的有一个方括号，后面又有一个，就是二维，四行 
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])       
    labels=['A','A','B','B']  
    return group,labels  
  
 #inX是你要输入的要新分类的“坐标”，dataSet是上面createDataSet的array，就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k   
def classify0(inX,dataSet,labels,k):   
    #dataSetSize是sataSet的行数，用上面的举例就是4,shape[0]表示矩阵第一维度的长度，也就是说行数。               
    dataSetSize=dataSet.shape[0]                       
    #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减
    diffMat=tile(inX,(dataSetSize,1))-dataSet         
    #上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方  
    sqDiffMat=diffMat**2               
    #axis=1是列相加，，这样得到了(x1-x2)^2+(y1-y2)^2               
    sqDistances=sqDiffMat.sum(axis=1)         
    #开根号，这个之后才是距离           
    distances=sqDistances**0.5   
    #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])                       
    sortedDistIndicies=distances.argsort()             
    
    classCount={}  
    for i in range(k):  
        voteIlabel=labels[sortedDistIndicies[i]]  
        #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1  
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1            
    #key=itemgerator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序=op  
    soredClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)   
    #返回类别最多的类别       
    return soredClassCount[0][0]              

#该函数的输人为文件名字符串 输出为训练样本矩阵和类标签向量(datingTestSet.txt)
def file2matrix(filename):
    fr = open(filename)
    #get the number of lines in the file
    numberOfLines = len(fr.readlines()) 
    #prepare matrix to return        
    returnMat = zeros((numberOfLines,3))  
    #prepare labels return         
    classLabelVector = []                       
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #去掉每行开头与结尾的符号的。因为在每行结尾都有个"/n"
        line = line.strip()
        #通过指定分隔符‘\t’对字符串进行切片
        listFromLine = line.split('\t')
        #存储
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        else: #listFromLine[-1] == 'didntLike'
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector
  
 #归一化特征值：newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    #参数0使得函数可以从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    #shape[0]表示矩阵第一维度的长度，也就是说行数。  
    m = dataSet.shape[0]
    #使用numpy库中tile()函数将变量内容复制成输人矩阵同样大小的矩阵
    normDataSet = dataSet - tile(minVals, (m,1))
    #注意这是具体特征值相除,对于某些数值处理软件包,/可能意味着矩阵除法.
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   

#测试算法：
def datingClassTest():
    #测试率：
    hoRatio = 0.10      #hold out 10%
    #从文件中读取数据
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    #转换为归一化特征值：数据集，范围，最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #数据集行数：
    m = normMat.shape[0]
    #用于测试的数据数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #每次都重新聚类，然后预测，返回结果：
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
#将一个32*32的图像矩阵转化为1*1024的向量。
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别系统测试代码：
def handwritingClassTest():
    hwLabels = []
    #获取目录的内容
    trainingFileList = listdir('trainingDigits')           #load the training set
    #文件数：
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        #文件名：“2_12.txt”
        fileNameStr = trainingFileList[i]
        #去掉.txt后缀：
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        #提取前面的真实数字2：
        classNumStr = int(fileStr.split('_')[0])
        #用来存放所有的分类标签：
        hwLabels.append(classNumStr)
        #最后将文件中的内容存储为1*1024的向量：
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    #读取测试文件夹：
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

# coding: utf-8

# In[11]:


from numpy import*
import operator


# In[12]:


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# In[24]:


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #shape[0]: 행 갯수, shape[1]: 열 갯수
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #tile(matrix, reps) -> reps가 (a, b) 튜플 형태인 경우, axb 행렬 만큼 matrix를 반복하여 새로운 행렬을 만든다.
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #행끼리 합 (row, column, depth) => 1: column끼리 
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #오름차순 정렬 후, 인덱스 값을 반환
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #키:값 투표(초기화)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # python3에서는 iteritems 대신 items.
    # iteritems는 key, value 값의 쌍을 반복자로 반환해줌
    #operator.itemgetter(0): 키로 정렬
    #operator.itemgetter(1): 값으로 정렬
    #sorted 함수는 정렬 후 리스트로 반환
    return sortedClassCount[0][0]    


# In[25]:


group, labels = createDataSet()


# In[26]:


print(group)


# In[18]:


print(labels)


# In[20]:


inX = array([1.0, 1.0])


# In[22]:


print(inX)


# In[27]:


classify0(inX, group, labels, 3)


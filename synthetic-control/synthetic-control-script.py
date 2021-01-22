#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install tslib')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install xlrd')


# In[2]:


import os
import sys
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

import tslib
from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.tests import testdata


# In[3]:


help(tslib)


# In[8]:


df = pd.read_csv('C:\Crazy in Study\AB Testing\Week 6\CVR.csv')


# In[9]:


df.set_index('Date')


# In[10]:


#.T横竖颠倒转换
df = df.set_index('Date').T


# In[15]:


df


# In[16]:


df.columns


# In[17]:


#把columns转化为日期格式
df.columns = pd.to_datetime(df.columns)


# In[18]:


df


# In[19]:


country = list(np.unique(df.index))

#将columns直接拉出来赋值给month
month = df.columns

#定义PredictKeyCountry
PredictKeyCountry = 'US'

#将PredictKeyCountry去除的部分，作为updated的country list
country.remove(PredictKeyCountry)

#将剩余的country list赋值给OtherCountry
OtherCountry = country

#分别定义MonthStart，MonthTrainEnd，MonthTestEnd
MonthStart = pd.to_datetime('2017-01-01')
MonthTrainEnd = pd.to_datetime('2018-12-01')
MonthTestEnd = pd.to_datetime('2019-12-01')


# In[20]:


#用循环分别给trainingMonth和testMonth赋值
trainingMonth = []

for i in pd.date_range(MonthStart, MonthTrainEnd, freq = 'MS'):
    trainingMonth.append(i) 

testMonth = []

for i in pd.date_range(MonthTrainEnd, MonthTestEnd, freq = 'MS'):
    testMonth.append(i) 


# In[21]:


trainingMonth


# In[22]:


trainDataMasterDict = {}
trainDataDict = {}
testDataDict = {}


# In[23]:


#对于其他国家
for key in OtherCountry:
    
    #从df里把相应key国家的data拉出来
    series = df.loc[key, ]
    
    #在dict的key里填入key，再把trainingMonth作为index以在从上面拉出来的series里找出相应月份数据，再将其填入dict里相应的key的后面
    trainDataMasterDict.update({key:series[trainingMonth].values})
    
    #这一步可以帮助看看 miss多少data是可以的，miss的data的percentage不一样时做出来的效果差异有多少，如果一点都不miss的话就如下填1
    #有的时候数据来源不可靠，或者miss了一些dimension的data，就可以用这一步（比如只填0.9）来看这些missing data的影响有多大
    (trainData,pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMasterDict[key]), 1)
    trainDataDict.update({key:trainData})
    
    #在dict的key里填入key，再把testMonth作为index以在从上面拉出来的series里找出相应月份数据，再将其填入dict里相应的key的后面
    testDataDict.update({key:series[testMonth].values})


# In[24]:


trainData


# In[25]:


trainDataMasterDict


# In[26]:


#对于参与测试的那一个国家再单独重复一遍上述过程
series = df.loc[PredictKeyCountry, ]
trainDataMasterDict.update({PredictKeyCountry:series[trainingMonth].values})
trainDataDict.update({PredictKeyCountry:series[trainingMonth].values})
testDataDict.update({PredictKeyCountry:series[testMonth].values})


# In[27]:


#把以上所有dict都转换成df
trainMasterDF = pd.DataFrame(trainDataMasterDict)
trainDF = pd.DataFrame(trainDataDict)
testDF = pd.DataFrame(testDataDict)


# In[28]:


trainDataMasterDict


# In[29]:


trainMasterDF


# In[30]:


trainDF


# In[31]:


testDF


# In[32]:


(U, s, Vh) = np.linalg.svd(trainDF-np.mean(trainDF))


# In[34]:


U


# In[33]:


s


# In[35]:


Vh


# In[36]:


s2 = np.power(s,2)
spectrum = np.cumsum(s2) / np.sum(s2)

plt.plot(spectrum)
plt.grid()
plt.title("Cumulative energy")
plt.figure()
plt.plot(s2)
plt.grid()
plt.xlabel("Ordered Singular Values") 
plt.ylabel("Energy")


plt.title("Singular Value Spectrum")

#这一步其实在画图看到底要拟合几个其他城市最好


# In[43]:


singvals = 2 #确定要用几个城市的数据来拟合，能拟合出最佳结果

#先做出model基本框架
#这个函数 是给它一个seriesToPredictKey和一个otherSeriesKeysArray，让它从后者中选出特定数目n个（singvals）以拟合出能拟合出最好的结果
#从而形成一个model的框架
#这里为什么用trainDF呢，因为就是要找地震发生前大家都正常的时候来尝试拟合，这样才可以拟合出地震实际发生时段的“如果没发生地震的话”的样子
rscModel = RobustSyntheticControl(PredictKeyCountry, singvals,len(trainDF),1,modelType='svd',
                                  svdMethod = 'numpy', otherSeriesKeysArray = OtherCountry)

#再拿我们的train期间数据去让这个model来fit 从而将它train、捏、塑造成想要的样子，好之后直接用它预测
rscModel.fit(trainDF)
denoiseDF = rscModel.model.denoisedDF()


# In[41]:


rscModel


# In[38]:


predictions = []

#往model里代入test阶段的othercountrys数据，做出prediction数据
predictions = np.dot(testDF[OtherCountry], rscModel.model.weights)
#这个model就是很nb 你把所有的other country series代进去它还是只选那 n个它觉得ok的，因为它黑箱，我们也不知道它拟合成啥样，所以只能代入所有
#让它自己选择

#直接拉出df中PredictKeyCountry的数据作为actual
actual = df.loc[PredictKeyCountry]

#往model里代入train阶段的othercountrys数据，做出前期可以与actual进行对比的fit数据
model_fit = np.dot(trainDF[OtherCountry][:], rscModel.model.weights)


# In[39]:


predictions


# In[131]:


fig, ax = plt.subplots(1,1)
tick_spacing = 35
# this is a bug in matplotlib
# label_markings = np.insert(years[::tick_spacing], 0, 'dummy')
label_markings = month.strftime('%y-%m')

ax.set_xticks(np.arange(len(label_markings)))
ax.set_xticklabels(label_markings, rotation = 90)
ax.set_xticklabels(label_markings)
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.plot(month, actual ,label = 'actual') #画出真实值
plt.xlabel('Month')
plt.ylabel('Visitor')
plt.plot(trainingMonth, model_fit, label='fitted model') #画出training阶段的fitted model
plt.plot(testMonth, predictions, label='counterfactual') #画出test阶段的预测线条
plt.title(PredictKeyCountry+', Singular Values used: '+str(singvals))

# xposition = pd.to_datetime(MonthTrainEnd,  errors='coerce')
plt.axvline(x=MonthTrainEnd, color='k', linestyle='--', linewidth=4)
# plt.grid()
plt.legend()


# In[133]:


#划重点，这种analysis所有的data都是actual的，只有这个model是我们强行基于actual data做出来的，没有什么真的预测的不知道的东西


# In[128]:


fig, ax = plt.subplots(1,1)
tick_spacing = 35
# this is a bug in matplotlib
# label_markings = np.insert(years[::tick_spacing], 0, 'dummy')
label_markings = month.strftime('%y-%m')

ax.set_xticks(np.arange(len(label_markings)))
ax.set_xticklabels(label_markings, rotation = 90)
ax.set_xticklabels(label_markings)
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.plot(month, actual ,label = 'actual') #画出真实值
plt.xlabel('Month')
plt.ylabel('Visitor')
plt.plot(trainingMonth, model_fit, label='fitted model') #画出training阶段的fitted model
plt.plot(testMonth, predictions, label='counterfactual') #画出test阶段的预测线条
plt.title(PredictKeyCountry+', Singular Values used: '+str(singvals))

# xposition = pd.to_datetime(MonthTrainEnd,  errors='coerce')
plt.axvline(x=MonthTrainEnd, color='k', linestyle='--', linewidth=4)
# plt.grid()
plt.legend()


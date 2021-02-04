#get all packages we need
!pip install matplotlib
!pip install tslib
!pip install sklearn
!pip install xlrd

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

#read data
df = pd.read_csv('./synthetic-control-data.csv')

#reset the index
df.set_index('Date')

#transpose df, let 'Date' become the columns
df = df.set_index('Date').T

#check out how df looks like now
df

#convert the columns('Date') to datetime format
df.columns = pd.to_datetime(df.columns)

#define country, month, PredictKeyCountry & OtherCountry
country = list(np.unique(df.index))
month = df.columns
PredictKeyCountry = 'US'
country.remove(PredictKeyCountry)
OtherCountry = country

#define MonthStart，MonthTrainEnd，MonthTestEnd
MonthStart = pd.to_datetime('2017-01-01')
MonthTrainEnd = pd.to_datetime('2018-12-01')
MonthTestEnd = pd.to_datetime('2019-12-01')

#definne trainingMonth & test Month by using a loop
trainingMonth = []

for i in pd.date_range(MonthStart, MonthTrainEnd, freq = 'MS'):
    trainingMonth.append(i) 

testMonth = []

for i in pd.date_range(MonthTrainEnd, MonthTestEnd, freq = 'MS'):
    testMonth.append(i) 

#define trainDataMasterDict, trainDataDict & testDataDict
trainDataMasterDict = {}
trainDataDict = {}
testDataDict = {}

for key in OtherCountry:
    
    #according to key, pull corresponding OtherCountry data from df as series
    series = df.loc[key, ]
    
    #fill in the keys & the values after the keys in trainDataMasterDict with series' value in trainingMonth
    trainDataMasterDict.update({key:series[trainingMonth].values})

    #fill in the keys & the values after the keys in testDataDict with series' value in testMonth
    testDataDict.update({key:series[testMonth].values})
    
    #this step is to check the influence of missing values (if any), e.g. if no any missing values, then fill 1
    (trainData,pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMasterDict[key]), 1)
	trainDataDict.update({key:trainData})


#fill in PredictKeyCountry's key & values in trainDataMasterDict, trainDataDict & testDataDict
series = df.loc[PredictKeyCountry, ]
trainDataMasterDict.update({PredictKeyCountry:series[trainingMonth].values})
trainDataDict.update({PredictKeyCountry:series[trainingMonth].values})
testDataDict.update({PredictKeyCountry:series[testMonth].values})

#convert all dicts to df format
trainMasterDF = pd.DataFrame(trainDataMasterDict)
trainDF = pd.DataFrame(trainDataDict)
testDF = pd.DataFrame(testDataDict)

#convert the data in df into a matrix we need in the future steps
(U, s, Vh) = np.linalg.svd(trainDF-np.mean(trainDF))

#plot the energy chart to see how many sigvals we should have to get the optimal result
s2 = np.power(s, 2)
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

#modeling
#build the model structure with determined amount of singvals and other basic info such as M, predictkey, etc.
#len(trainDF) is the length of training duration in specific units e.g.in this case are 24 months
singvals = 2
rscModel = RobustSyntheticControl(PredictKeyCountry, singvals, len(trainDF), 1, modelType='svd', svdMethod = 'numpy', otherSeriesKeysArray = OtherCountry) 

#fit the model with training data
rscModel.fit(trainDF)
denoiseDF = rscModel.model.denoisedDF()

#output the data we need by inputting correspoding data into the model
#the model is like a black box, it chose 2 other countries, but we don't know which 2, so we just need to input all OtherCountry and let it choose again
actual = df.loc[PredictKeyCountry]
model_fit = np.dot(trainDF[OtherCountry][:], rscModel.model.weights) 
predictions = []
predictions = np.dot(testDF[OtherCountry], rscModel.model.weights)

#plot all outputs to see how the model performs
fig, ax = plt.subplots(1,1)
tick_spacing = 35
# this is a bug in matplotlib
# label_markings = np.insert(years[::tick_spacing], 0, 'dummy')
label_markings = month.strftime('%y-%m')

ax.set_xticks(np.arange(len(label_markings)))
ax.set_xticklabels(label_markings, rotation = 90)
ax.set_xticklabels(label_markings)
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.plot(month, actual ,label = 'actual') #plot the actual data
plt.xlabel('Month')
plt.ylabel('Visitor')
plt.plot(trainingMonth, model_fit, label='fitted model') #plot trainingMonth fitted model
plt.plot(testMonth, predictions, label='counterfactual') #plot testMonth prediction
plt.title(PredictKeyCountry+', Singular Values used: '+str(singvals))

# xposition = pd.to_datetime(MonthTrainEnd,  errors='coerce')
plt.axvline(x=MonthTrainEnd, color='k', linestyle='--', linewidth=4)
# plt.grid()
plt.legend()
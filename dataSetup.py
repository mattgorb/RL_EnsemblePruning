import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,LSTM,GRU
import pywt
from collections import defaultdict
from keras import losses
from sklearn import metrics
import math
import sys

def dataSetup(file='sp500.csv',multivariate=False,testDate=0,trainingIteration=0,time_steps_back=75, printDates=False):

    df = pd.read_csv(file, sep=',')

    if(printDates):
        df['Ntime'] = df['Ntime'].apply(str)
        df['Ntime']=pd.to_datetime(df['Ntime']).dt.strftime('%m/%d/%Y')

        print("Training Dates:")
        print(df['Ntime'].values[trainingIteration*62+time_steps_back*2+testDate]+" to "+df['Ntime'].values[trainingIteration*62+time_steps_back*2+testDate+252*5-1]+"\n")

        print("Validation Dates:")
        print(df['Ntime'].values[(trainingIteration*62+252*5+time_steps_back*2+testDate)]+" to "+df['Ntime'].values[(trainingIteration*62+252*5+62*2+time_steps_back*2+testDate)-1]+"\n")


        print("Test Date:")
        print(df['Ntime'].values[(trainingIteration*62+252*5+62*2+time_steps_back*2+testDate)]+"\n")

    testValue=df['Close Price'].values[(trainingIteration*62+252*5+62*2+time_steps_back*2+testDate)]

    unscaledData={}
    for var in range(2,len(df.columns[2:])+2):
        unscaledData[df.columns[var]]=df[df.columns[var]].values[trainingIteration*62:(trainingIteration*62+252*5+62*2+time_steps_back*2+testDate)]
    
    cA_list=dict.fromkeys(unscaledData.keys(),[])
    cD_list=dict.fromkeys(unscaledData.keys(),[])
    for key, value in unscaledData.items():
        cA_list[key],cD_list[key]=pywt.dwt(value, 'haar')

    scalers=dict.fromkeys(unscaledData.keys(),[])
    scaledData=dict.fromkeys(unscaledData.keys(),[])
    for key, value in cA_list.items():
        scalers[key] = MinMaxScaler(feature_range=(-1, 1))
        scaledData[key]=scalers[key].fit_transform(np.reshape(value,(value.shape[0],1)))

    a=0
    trainingData = defaultdict(list)
    
    while a<len(scaledData['Close Price'])-time_steps_back:
        for key, value in scaledData.items():
            trainingData[key].append(value[a:a+time_steps_back])
        a+=1

    testDataDict = defaultdict(list)
    if(multivariate):
        for key, value in scaledData.items():
            testDataDict[key].append(value[a:a+time_steps_back])
    else:
        testDataDict['Close Price'].append(scaledData['Close Price'][a:a+time_steps_back])


    testData=np.array(testDataDict['Close Price'])
    if(multivariate):
        for key, value in testDataDict.items():
            if(key=='Close Price'):
                continue
            testData=np.concatenate((testData, value), axis=2)    

    X=np.array(trainingData['Close Price'])
    Y=np.reshape(np.array(scaledData['Close Price']),(np.array(scaledData['Close Price']).shape[0]))[time_steps_back:]
    Y=np.reshape(Y, (Y.shape[0],1))    

    if(multivariate):
        for key, value in trainingData.items():
            if(key=='Close Price'):
                continue
            X=np.concatenate((X, value), axis=2)

    return X,Y,testData,testValue,scalers['Close Price'],cA_list['Close Price'][time_steps_back:],cD_list['Close Price'][time_steps_back:]








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
from collections import OrderedDict
import operator
import copy

from dataSetup import *
from models import *

time_steps_back=150
validation_size=31*3


def model_stack():
    multivariate_models={
        "multi_lstm_512":multivariateLSTM(512,'mse',time_steps_back),
        "multi_lstm_128":multivariateLSTM(1024,'mse',time_steps_back),
        "multi_gru_512":multivariateGRU(512,'mse',time_steps_back),
        "multi_gru_256":multivariateGRU(256,'mse',time_steps_back),
        "multi_gru_2":multivariateGRU(700,'mse',time_steps_back),
        "multi_gru_3":multivariateGRU(400,'mse',time_steps_back),
        "multi_gru_1024":multivariateGRU(1024,'mse',time_steps_back),
        "multi_gru_4":multivariateGRU(300,'mse',time_steps_back),
        "multi_lstm_3":multivariateLSTM(400,'mse',time_steps_back),
        "multi_lstm_4":multivariateLSTM(800,'mse',time_steps_back),
        "multi_lstm_5":multivariateLSTM(1300,'mse',time_steps_back),
        "multi_lstm_6":multivariateLSTM(550,'mse',time_steps_back)       
        }
    singlevariate_models={
        "single_GRU_64":singleGRU(64,'mse',time_steps_back),
        "single_GRU_32":singleGRU(128,'mse',time_steps_back)}

    multivariate_models = OrderedDict(sorted(multivariate_models.items(), key=operator.itemgetter(0)))
    singlevariate_models = OrderedDict(sorted(singlevariate_models.items(), key=operator.itemgetter(0)))
    return multivariate_models,singlevariate_models



def train_all(trainingIteration):
    batch=20
    epochs=100
    val_losses=[]
    for key, value in multivariate_models.items():
        weight_file=key+'_'+str(trainingIteration)
        model,lowest_loss=train(value,weight_file,trainX,trainY,valX,valY,batch,epochs)
        val_losses.append(lowest_loss)
    for key, value in singlevariate_models.items():
        weight_file=key+'_'+str(trainingIteration)
        model,lowest_loss=train(value,weight_file,trainX_single,trainY_single,valX_single,valY_single,batch,epochs)        
        val_losses.append(lowest_loss)
        
    #remove bad models.
    idx = [i for i, x in enumerate(val_losses) if x in sorted(val_losses)[:10]]

    newModels_multi={}
    newModels_single={}
    for i in idx:
        if(i<len(multivariate_models)):
                key=list(multivariate_models)[i]
                newModels_multi[key]=multivariate_models[key]
        else:
                key=list(singlevariate_models)[i-len(list(multivariate_models))]
                newModels_single[key]=singlevariate_models[key]                

    return newModels_multi,newModels_single
        
def printLosses(losses,actual,printPersistance=False):
    for key, value in losses.items():
        mae=metrics.mean_absolute_error(np.array(value), np.array(actual))
        mape=mean_absolute_percentage_error(np.array(value), np.array(actual))
        print(key)
        print("MAE,MAPE:")
        print(mae)
        print(mape)
        print('\n')

    if(printPersistance):
        print("Persistance:")
        mse=metrics.mean_squared_error(np.array(actual[:-1]), np.array(actual[1:]))
        mae=metrics.mean_absolute_error(np.array(actual[:-1]), np.array(actual[1:]))
        mape=mean_absolute_percentage_error(np.array(actual[:-1]), np.array(actual[1:]))
        print(mae)
        print(mape)




def individual_model_baseline(trainingIteration,plot=False,printModelLoss=True):
    predictions_multivariate = defaultdict(list)
    predictions_single = defaultdict(list)

    for key, value in multivariate_models.items():
        predictions_multivariate[key]=[]

    for key, value in singlevariate_models.items():
        predictions_single[key]=[]

    X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,time_steps_back=time_steps_back,printDates=True)
    X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back)

    actual=[]
    for i in range(1,63):
        actual.append(testValue)

        for key, value in multivariate_models.items():
            model=value
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            y=scalers.inverse_transform(model.predict(testData))
            reconstruct=pywt.idwt(np.reshape(y, (y.shape[0])), cD_list[-1], 'haar')
            predictions_multivariate[key].append(reconstruct[0])

        for key, value in singlevariate_models.items():
            model=value
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            y=scalers.inverse_transform(model.predict(testData_single))
            reconstruct=pywt.idwt(np.reshape(y, (y.shape[0])), cD_list[-1], 'haar')
            predictions_single[key].append(reconstruct[0])
        

        X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,testDate=i,time_steps_back=time_steps_back)
        X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back,testDate=i)

    if(printModelLoss,actual):
        printLosses(predictions_single,actual)
        printLosses(predictions_multivariate,actual,printPersistance=True)


    if(plot):
            figure = plt.figure()
            tick_plot = figure.add_subplot(1, 1, 1)
            tick_plot.plot(np.arange(0,len(actual)), actual, marker='o', label='actual')
            for key, value in predictions_multivariate.items():
                tick_plot.plot(np.arange(0,len(actual)), value, marker='o', label=key)
            for key, value in predictions_single.items():
                tick_plot.plot(np.arange(0,len(actual)), value, marker='o', label=key)
            plt.legend(loc='upper left')
            plt.show()




def addRewardData(trainingIteration):
    gamma=20
    ensemble_X=[]
    ensemble_Y=[]
    num_models=len(multivariate_models)+len(singlevariate_models)
    for i in range(len(trainX)):
        addLine=[]
        online_multiVar=np.reshape(trainX[i], (1,trainX[i].shape[0],trainX[i].shape[1]))

        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_multiVar)[0][0]
            reward=1/(1+abs(pred-trainY[i]))
            reward=reward**gamma
            addLine.append(reward)

        online_singleVar=np.reshape(trainX_single[i], (1,trainX_single[i].shape[0],1))

        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_singleVar)[0][0]
            reward=1/(1+abs(pred-trainY[i]))
            reward=reward**gamma
            addLine.append(reward)
            
        addLine=np.reshape(addLine, (1,num_models))

        ensemble_X.append(online_multiVar)
        ensemble_Y.append(addLine)
    for i in range(len(valX)):
        addLine=[]
        online_multiVar=np.reshape(valX[i], (1,valX[i].shape[0],valX[i].shape[1]))

        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_multiVar)[0][0]
            reward=1/(1+abs(pred-valY[i]))
            reward=reward**gamma
            addLine.append(reward)

        online_singleVar=np.reshape(valX_single[i], (1,valX_single[i].shape[0],1))

        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_singleVar)[0][0]
            reward=1/(1+abs(pred-valY[i]))
            reward=reward**gamma
            addLine.append(reward)
            
        addLine=np.reshape(addLine, (1,num_models))

        ensemble_X.append(online_multiVar)
        ensemble_Y.append(addLine)

    ensemble_X=np.array(ensemble_X)
    ensemble_Y=np.array(ensemble_Y)
        
    ensemble_X=np.reshape(ensemble_X, (ensemble_X.shape[0],ensemble_X.shape[2],ensemble_X.shape[3]))
    ensemble_Y=np.reshape(ensemble_Y, (ensemble_Y.shape[0],ensemble_Y.shape[2]))
    return ensemble_X,ensemble_Y


def RewardStretchedEnsemble(trainingIteration,fit=True):
    X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,time_steps_back=time_steps_back,printDates=True)
    trainX,valX,trainY,valY=X[:-validation_size],X[-validation_size:],Y[:-validation_size],Y[-validation_size:]

    X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back)
    trainX_single,valX_single,trainY_single,valY_single=X[:-validation_size],X[-validation_size:],Y[:-validation_size],Y[-validation_size:]

    gamma=20
    num_models=len(multivariate_models)+len(singlevariate_models)
    
    ensembleModel=Ensemble_Network(32,'mse',num_models)

    ensemble_X,ensemble_Y=addRewardData(trainingIteration)
    print("Training ensemble...")
    if(fit):
        ensembleModel.fit(ensemble_X, ensemble_Y, epochs=25, batch_size=10, verbose=0)
    else:
        ensembleModel.load_weights('ensemble.h5')
    ensembleModel.save_weights('ensemble.h5')

    actual=[]
    preds=[]
    for i in range(1,63):
        actual.append(testValue)

        idx = np.random.choice(np.arange(len(ensemble_Y)), 20, replace=False)
    
        x_sample = ensemble_X[idx]
        y_sample = ensemble_Y[idx]
        
        ensembleModel.fit(x_sample, y_sample, epochs=5, batch_size=5, verbose=0)

        ensemble_prediction=ensembleModel.predict(testData)
        print(ensemble_prediction)
        best=np.argmax(ensemble_prediction)

        ensemble_X=np.concatenate((ensemble_X, testData))


        addLine=[]
        losses=[]
        j=0

        cD_list_old=cD_list
        scalers_old=scalers

        X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,testDate=i,time_steps_back=time_steps_back)
        X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back,testDate=i)
        
        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(testData)
            if(j==best):
                ensemblePred=pred
            loss=abs(pred-Y[-1])
            losses.append(loss)
            reward=1/(1+abs(pred-Y[-1]))
            reward=reward**gamma
            addLine.append(reward)
            j+=1


        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+'.h5')
            pred=value.predict(testData_single)
            if(j==best):
                ensemblePred=pred
            loss=abs(pred-Y[-1])
            losses.append(loss)
            reward=1/(1+abs(pred-Y[-1]))
            reward=reward**gamma
            addLine.append(reward)
            j+=1

        y=scalers_old.inverse_transform(ensemblePred)
        reconstruct=pywt.idwt(np.reshape(y, (y.shape[0])), cD_list_old[-1], 'haar')
        preds.append(reconstruct[-1])

        addLine=np.reshape(addLine, (1,num_models))

        ensemble_Y=np.concatenate((ensemble_Y, addLine))            

    mae=metrics.mean_absolute_error(np.array(preds), np.array(actual))

    print("Training iteration MAE")
    print(mae)
    return preds, actual






def StandardModelStacking(trainingIteration,fit=True):
    num_models=len(multivariate_models)+len(singlevariate_models)
    ensemble_X=[]
    ensemble_Y=[]
    num_models=len(multivariate_models)+len(singlevariate_models)
    for i in range(len(trainX)):
        addLine=[]
        online_multiVar=np.reshape(trainX[i], (1,trainX[i].shape[0],trainX[i].shape[1]))

        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_multiVar)[0][0]
            addLine.append(pred)

        online_singleVar=np.reshape(trainX_single[i], (1,trainX_single[i].shape[0],1))

        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_singleVar)[0][0]
            addLine.append(pred)
            
        addLine=np.reshape(addLine, (1,num_models))

        ensemble_Y.append(trainY[i])
        ensemble_X.append(addLine)
    for i in range(len(valX)):
        addLine=[]
        online_multiVar=np.reshape(valX[i], (1,valX[i].shape[0],valX[i].shape[1]))

        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_multiVar)[0][0]

            addLine.append(pred)

        online_singleVar=np.reshape(valX_single[i], (1,valX_single[i].shape[0],1))

        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(online_singleVar)[0][0]

            addLine.append(pred)
            
        addLine=np.reshape(addLine, (1,num_models))

        ensemble_X.append(addLine)
        ensemble_Y.append(valY[i])

    ensemble_X=np.array(ensemble_X)
    ensemble_Y=np.array(ensemble_Y)



    ensembleModel = Sequential()
    ensembleModel.add(GRU(32, input_shape=(1,num_models)))
    ensembleModel.add(Dense(1))
    
    ensembleModel.compile(loss='mse', optimizer='adam')
    

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=0)                                  
    checkpointer=ModelCheckpoint('weights/stack.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    earlystopper=EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5, verbose=0, mode='auto')
    
    ensembleModel.fit(ensemble_X[:-60], ensemble_Y[:-60], validation_data=(ensemble_X[-60:], ensemble_Y[-60:]),epochs=100, batch_size=20, verbose=2, shuffle=True,callbacks=[checkpointer,earlystopper,reduce_lr])

    #nsembleModel.fit(ensemble_X, ensemble_Y, , validation_data=(valX, valY),epochs=100, batch_size=10, verbose=2,shuffle=True)

    addLine=[]



    actual=[]
    preds=[]

    X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,testDate=0,time_steps_back=time_steps_back)
    X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back,testDate=i)
        

    for i in range(1,63):
        actual.append(testValue)

        idx = np.random.choice(np.arange(len(ensemble_Y)), 20, replace=False)
    
        x_sample = ensemble_X[idx]
        y_sample = ensemble_Y[idx]

        addLine=[]
        for key, value in multivariate_models.items():
            value.load_weights("weights/"+key+"_"+str(trainingIteration)+'.h5')
            pred=value.predict(testData)
            addLine.append(pred)


        for key, value in singlevariate_models.items():
            value.load_weights("weights/"+key+"_"+'.h5')
            pred=value.predict(testData_single)
            addLine.append(pred)


        addLine=np.reshape(addLine, (1,1,num_models))
        ensemble_X=np.concatenate((ensemble_X, addLine))

        
        ensembleModel.fit(x_sample, y_sample, epochs=5, batch_size=5, verbose=0)
        ensemble_prediction=ensembleModel.predict(addLine)[0][0]

        cD_list_old=cD_list
        scalers_old=scalers

        X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,testDate=i,time_steps_back=time_steps_back)
        X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back,testDate=i)

        y=scalers_old.inverse_transform(ensemble_prediction)
        reconstruct=pywt.idwt(np.reshape(y, (y.shape[0])), cD_list_old[-1], 'haar')
        preds.append(reconstruct[-1])

        #y=scalers.inverse_transform(Y[-1])
        #reconstruct=pywt.idwt(np.reshape(y, (y.shape[0])), cD_list[-1], 'haar')
        y=np.array(Y[-1])
        y=np.reshape(y, (y.shape[0],1))
        ensemble_Y=np.concatenate((ensemble_Y,y))
        


    mae=metrics.mean_absolute_error(np.array(preds), np.array(actual))

    print("Training iteration MAE")
    print(mae)

    return preds,actual    





preds=[]
actual=[]
for trainingIteration in range(6):

    
    multivariate_models,singlevariate_models=model_stack()

    X,Y,testData,testValue,scalers,cA_list,cD_list=dataSetup(multivariate=True,time_steps_back=time_steps_back,printDates=False,trainingIteration=trainingIteration)
    trainX,valX,trainY,valY=X[:-validation_size],X[-validation_size:],Y[:-validation_size],Y[-validation_size:]

    X,Y,testData_single,testValue,scalers,cA_list,cD_list=dataSetup(time_steps_back=time_steps_back,trainingIteration=trainingIteration)
    trainX_single,valX_single,trainY_single,valY_single=X[:-validation_size],X[-validation_size:],Y[:-validation_size],Y[-validation_size:]

    multivariate_models,singlevariate_models=train_all(trainingIteration)

    individual_model_baseline(trainingIteration,plot=False)
    p,a=RewardStretchedEnsemble(trainingIteration)
    
    preds.extend(p)
    actual.extend(a)

mae=metrics.mean_absolute_error(np.array(preds), np.array(actual))

print(mae)


print('persistance')
print(metrics.mean_absolute_error(np.array(actual[:-1]), np.array(actual[1:])))
    
figure = plt.figure()
tick_plot = figure.add_subplot(1, 1, 1)
tick_plot.plot(np.arange(0,len(actual)), actual, marker='o', label='actual')
tick_plot.plot(np.arange(0,len(actual)), preds, marker='o', label='guesses')
plt.show()
    







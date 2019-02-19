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
from keras.callbacks import History 

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


def multivariateGRU(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(time_steps_back,19)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def multivariateLSTM(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(time_steps_back,19)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def multivariateLSTM_sequence(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(time_steps_back,19),return_sequences=True))
    model.add(LSTM(nodes, input_shape=(time_steps_back,19)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def singleLSTM(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(time_steps_back,1)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def singleLSTM_sequences(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(time_steps_back,1),return_sequences=True))
    model.add(LSTM(nodes, input_shape=(time_steps_back,1)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def singleGRU(nodes, loss,time_steps_back):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(time_steps_back,1)))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')
    return model

def Ensemble_Network(nodes,loss,num_models):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(150,19)))
    model.add(Dense(num_models))
    model.compile(loss=loss, optimizer='adam')
    return model



def train(model,weights_file,x,y,valX,valY,batch,epochs):
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=0)                                  
    checkpointer=ModelCheckpoint('weights/'+weights_file+'.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    earlystopper=EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5, verbose=0, mode='auto')
    model.fit(x, y, validation_data=(valX, valY),epochs=epochs, batch_size=batch, verbose=0, shuffle=True,callbacks=[checkpointer, history,earlystopper,reduce_lr])
    lowest_val_loss=min(history.history['val_loss'])
    print(lowest_val_loss)
    return model,lowest_val_loss

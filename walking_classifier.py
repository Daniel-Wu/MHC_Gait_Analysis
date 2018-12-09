# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:18:13 2018

Walking Classifier
A simple 1D Convolutional Neural Network for classifying whether
data is walking or not. 

Great resource:
https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

@author: dwubu
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn import metrics
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

# =============================================================================
# Import File Data and assemble dataset
# =============================================================================

def makeDatasets(window_directory_str):
    '''makeDatasets
    Reads in files to a dataset
    Returns a single dataframe containing all x, y, and z windows of the files
    '''
    all_data = []

    for filename in os.listdir(window_directory_str):
        
        window_directory = os.fsencode(window_directory_str)
        
        #Read in the hdf of each file, add it to the dataframe
        filepath = os.path.join(window_directory.decode(), filename)
        data = pd.read_hdf(filepath)
        
        for index, series in data.iterrows():
            window = [series['xwindows'], series['ywindows'], series['zwindows']]
        
            all_data.append(window)
        
    return np.swapaxes(np.array(all_data), 1, 2) #pd.concat(all_data)


# =============================================================================
# Load in the data
# =============================================================================

#Get the walk data
data_dir = 'C:/Users/dwubu/Desktop/6mwtInhouseFiltered/Walk'
walk_data = makeDatasets(data_dir)
data_dir = 'C:/Users/dwubu/Desktop/6mwtInhouseFiltered/Rest'
rest_data = makeDatasets(data_dir)

x_train = np.concatenate((walk_data, rest_data), axis=0)
y_train = np.concatenate((np.array([1] * len(walk_data)), np.array([0] * len(rest_data))))

# =============================================================================
# Machine Learning Model        
# =============================================================================
 

model = Sequential()
# ENTRY LAYER
#model.add(Input(shape=(200,3)))
model.add(Conv1D(100, 10, activation='relu', input_shape=(200, 3)))
model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160, 10, activation='relu'))
model.add(Conv1D(160, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))#activation='softmax'))
print(model.summary())


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

BATCH_SIZE = 400
EPOCHS = 10

model.fit(x_train,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0)
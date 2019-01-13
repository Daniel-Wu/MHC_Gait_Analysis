# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:42:42 2018

Python script to be run on the sherlock cluster, which iterates through all 
the MHC 2.0 6 minute walk test dataset, and trains a model on it

Precondition: reads from an hdf5 file containing a 'data' group of shape
(num_samples, window_length, 3) containing all the windows

@author: Daniel Wu
"""
import os
import numpy as np
import pandas as pd
import h5py
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D


#Contains data for walk tests
directory = "/scratch/users/danjwu/6mwt_windows/data_windows.hdf5"

            
# =============================================================================
# Data generator
# =============================================================================
class SixMWTSequence(keras.utils.Sequence):
    '''
    SixMWTSequence
    Extends keras inbuilt sequence to create a data generator
    Saves on RAM by loading data from hdf5 files in memory
    __del__ way of closing files isn't great - find a better way sometime
    '''
    def __init__(self, walk_data_path, rest_data_path, batch_size):
        #Open up files
        self.walk_data = h5py.File(walk_data_path, 'r')['data']
        self.rest_data = h5py.File(rest_data_path, 'r')['data']
        #Label walking as 1, resting as 0
        self.labels = np.array([1]*self.walk_data.shape[0] + [0]*self.rest_data.shape[0])
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil((self.walk_data.shape[0] + self.rest_data.shape[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        #Figure out which file it's in
        if (idx + 1)*self.batch_size < self.walk_data.shape[0]:
            #Entire batch in first file
            batch_x = self.walk_data[idx * self.batch_size:(idx + 1) * self.batch_size]
            
        elif idx*self.batch_size >= self.walk_data.shape[0]:
            #Entire batch in second file
            idx -= self.walk_data.shape[0]
            batch_x = self.rest_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            #Mix and match
            batch_x = np.concatenate((self.walk_data[idx * self.batch_size : ],
                                      self.walk_data[0 : (idx + 1)*self.batch_size - self.walk_data.shape[0]]))
        
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
#    def __del__(self):
#        self.walk_data.close()
#        self.rest_data.close()
    

# =============================================================================
# Machine Learning Model        
# =============================================================================
 

model = Sequential()
# ENTRY LAYER
#model.add(Input(shape=(200,3)))
model.add(Conv1D(100, 20, activation='relu', input_shape=(200, 3)))
model.add(Conv1D(100, 20, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160, 20, activation='relu'))
model.add(Conv1D(160, 20, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#activation='softmax'))
print(model.summary())


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# =============================================================================
# Training the model
# =============================================================================
output_dir = r"C:\Users\dwubu\Desktop" #"/scratch/users/danjwu/results"
walk_data_file = r"C:\Users\dwubu\Desktop\subset_data\data_windows.hdf5"#"/scratch/users/danjwu/6mwt_windows/data_windows.hdf5"
rest_data_file = r"C:\Users\dwubu\Desktop\subset_data\data_windows_rest.hdf5"#"/scratch/users/danjwu/6mwt_windows/data_windows_rest.hdf5"
batch_size = 16

#training_batch_generator = SixMWTSequence(walk_data_file, rest_data_file, batch_size)
##VALIDATION IS TODO
##validation_batch_generator = SixMWTSequence(validation_filenames, validation_labels, batch_size)
#
#num_training_samples = len(training_batch_generator)
#num_validation_samples = 0
#num_epochs = 20
#
#model.fit_generator(generator=training_batch_generator,
#                    steps_per_epoch=(num_training_samples // batch_size),
#                    epochs=num_epochs,
#                    verbose=1,
#                    #validation_data=validation_batch_generator,
#                    #validation_steps=(num_validation_samples // batch_size),
#                    use_multiprocessing=True,
#                    workers=16,
#                    max_queue_size=32)

### Version with all data loaded into memory
with h5py.File(walk_data_file, 'r') as walk_file:
    with h5py.File(rest_data_file, 'r') as rest_file:
        x_train = np.concatenate((walk_file['data'][:], rest_file['data'][:]), axis=0)
        y_train = np.concatenate((np.array([1] * len(walk_file['data'][:])), np.array([0] * len(rest_file['data'][:]))))

BATCH_SIZE = 32
EPOCHS = 20

#Currently arbitrarily taking 521 walk points and 100 rest points as validation
#This is a bad split, but lazy
history = model.fit(x_train[521:-100],
                    y_train[521:-100],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0,
                    validation_data=(np.concatenate((x_train[:521], x_train[-100:]), axis = 0), 
                                     np.concatenate((y_train[:521], y_train[-100:]), axis = 0)))

model.save(os.path.join(output_dir, "model.h5"))
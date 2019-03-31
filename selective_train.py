# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:26:35 2019

Python script to be run on the sherlock cluster, which iterates through selected healthcodes in 
the MHC 2.0 6 minute walk test dataset, and trains a model on it

Precondition: reads from an hdf5 file containing groups labeled by healthcode of shape
(num_samples, window_length, 3) containing all the windows from that healthcode.

Run preprocess_data.py on the walk_data,
Change output_dir, data_file vars below to point at those files

TODO:
    Make this work with local filepaths
    Rewrite the generator
    Rewrite split data

@author: Daniel Wu
"""
import os
import platform
import numpy as np
import pandas as pd
import h5py
import keras
import threading
import math
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard


#True when on sherlock (or any other linux system)
on_sherlock = platform.system() == 'Linux'

# =============================================================================
# PARAMETERS
# =============================================================================
validation_split = 0.4
labels_of_interest = ["heartCondition"]

#File locations
if(on_sherlock):
    output_dir = "/scratch/PI/euan/projects/mhc/code/daniel_code/results"
    data_file = "/scratch/PI/euan/projects/mhc/code/daniel_code/data_windows_walk.hdf5"
    label_table_file = "/scratch/PI/euan/projects/mhc/code/daniel_code/combined_health_label_table.pkl"
else:
    output_dir = r"C:\Users\dwubu\Desktop"
    data_file = r"C:\Users\dwubu\Desktop\subset_data\data_windows.hdf5"

#Training metrics
if(on_sherlock):
    from concise.metrics import tpr, tnr, fpr, fnr, precision, f1
    model_metrics = ['accuracy',tpr,tnr,fpr,fnr,precision,f1]
else:
    model_metrics = ['accuracy']

#Training parameters
if(on_sherlock):
    batch_size = 256
    canMultiprocess = False
else:
    batch_size = 32
    canMultiprocess = False

# =============================================================================
# Extract only the needed data into a file
# =============================================================================

def extract_labels(labels = ['heartCondition'], label_table_path = label_table_file):
    '''
    Returns a dataframe indexed by healthCodes with columns of requested labels
    taken from the label table file
    '''
    label_df = pickle.load(open(label_table_path, 'rb')) 
    label_df = label_df[labels]
    return label_df.dropna()
        

def extract_records(healthCodes, data_path, out_path):
    '''
    Extracts the all of the healthcodes in healthCodes
    from the data_file, aggregates and ignores healthcodes, 
    and saves to a new hdf5 file.
    '''
    
    #Complete file
    data_file = h5py.File(data_path, 'r')
    
    #The new file
    new_file = h5py.File(out_path, 'a')
    
    #Go through the keys of the old file 
    valid_codes = set(healthCodes)
    for code in data_file.keys():
        #If the healthcode is valid, copy into new file
        if code in valid_codes:
            data_file.copy(code, new_file)
    
    #Return the HDF file
    return new_file

def extract_data(data_file):
    """
    Wrapper function that extracts the labelled data from the 6mwt results
    and splits the data
    
    Returns an HDF5 file containing all the filtered data
    and a dataframe containing the labels for the healthcodes inside
    
    """
    #Get labels
    label_df = extract_labels(labels = labels_of_interest, label_table_path = label_table_file)
    healthCodes = label_df.index.tolist()
    
    #Make a temporary data file in the same folder
    out_path = os.path.dirname(data_file) + os.sep + "filtered_data.hdf5"
    #Extract the records out from the 
    filtered_data = extract_records(healthCodes, data_file, out_path)
    
    return filtered_data, label_df

# =============================================================================
# Split data files into validation and test
# =============================================================================

def split_data(file_path, split, new_folder):
    '''
    split_data(file_path, split)
    splits the data in the 'data' dataset of the hdf file at filepath
    with the given split ratio, between 0 and 1.
    Uses new_folder in the new filepath
    Returns a tuple with two filenames, the first with (1-split), the second with (split) percent of the data
    '''
    with h5py.File(file_path, 'r') as hdf_file:
        data = hdf_file['data']
        (num_samples, window_len, channels) = data.shape
        
        split_num = math.floor(num_samples * split)
        
        #Open and save the valdation set
        out_dir = os.path.join(os.path.dirname(file_path), new_folder)
        validation_path = os.path.join(out_dir, "validation.hdf5")
        
        with h5py.File(validation_path, 'w') as validation_file:
            validation_file.create_dataset('data', 
                                           (split_num, 
                                           window_len, 
                                           channels)
                                          )
                                           
            #validation_file['data'][:] = data[:split_num]
            for i in range(split_num):
                validation_file['data'][i] = data[i]
        
        #Open and save the test set
        test_path = os.path.join(out_dir, "test.hdf5")
        
        with h5py.File(test_path, 'w') as test_file:
            test_file.create_dataset('data', 
                                      (num_samples - split_num, 
                                      window_len, 
                                      channels)
                                     )
            #test_file['data'][:] = data[split_num : ]
            for i in range(split_num, num_samples):
                test_file['data'][i - split_num] = data[i]
                
        
        return (test_path, validation_path, split_num)
        
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
    def __init__(self, walk_data_path, batch_size, label_df):
        #Open up files
        self.lock = threading.Lock()
        self.walk_file = h5py.File(walk_data_path, 'r')
        self.walk_data = self.walk_file['data']
        self.labels = label_df
        
        self.batch_size = batch_size
        
        if(balanceDataset):
            #Enforce a 50/50 split in each batch
            self.num_rest_points = int(self.batch_size / 2)
            self.num_walk_points = self.batch_size - self.num_rest_points
        else:
            #Find the number of walk_data_points per batch, proportional to total data points
            self.num_walk_points = int((self.walk_data.shape[0]/(self.walk_data.shape[0] + self.rest_data.shape[0])) * self.batch_size)
            self.num_rest_points = self.batch_size - self.num_walk_points


    def __len__(self):
        #Find how many batches fit in our dataset
        #This "crops" out a couple datapoints not divisible by the batch at the end
        return min(
                int(self.walk_data.shape[0]/self.num_walk_points),
                int(self.rest_data.shape[0]/self.num_rest_points))

    def __getitem__(self, idx):
        
        with self.lock:
            #Grab the batch members
            batch_x = np.concatenate((self.walk_data[idx * self.num_walk_points : (idx + 1) * self.num_walk_points], 
                                      self.rest_data[idx * self.num_rest_points : (idx + 1) * self.num_rest_points]))
    
            #Generate the labels
            batch_y = np.concatenate(([1]*self.num_walk_points, [0]*self.num_rest_points))
            
            return batch_x, batch_y
        
    def __del__(self):
        self.walk_file.close()    

# =============================================================================
# Defining the CNN
# =============================================================================

model = Sequential()
# ENTRY LAYER
model.add(Conv1D(100, 20, activation='relu', input_shape=(200, 3)))
model.add(BatchNormalization())

#model.add(Conv1D(100, 20, activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling1D(3))

#model.add(Conv1D(160, 20, activation='relu'))
#model.add(BatchNormalization())

model.add(Conv1D(160, 20, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))
model.add(Dense(40, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))
print(model.summary())

#Loss function - taken from kerasAC.custom_losses  -   need to figure out weights before using
#def get_weighted_binary_crossentropy(w0_weights, w1_weights):
#    import keras.backend as K
#    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
#    # In addition, weight everything with label -1 to 0
#    w0_weights=np.array(w0_weights);
#    w1_weights=np.array(w1_weights);
#    thresh=-0.5
#
#    def weighted_binary_crossentropy(y_true,y_pred):
#        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
#        nonAmbig = K.cast((y_true > -0.5),'float32')
#        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
#        return K.mean(K.binary_crossentropy(y_pred, y_true)*nonAmbigTimesWeightsPerTask, axis=-1);
#    return weighted_binary_crossentropy; 

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=model_metrics)

# =============================================================================
# Training the model
# =============================================================================

#Split the dataset, if data not already split
if(on_sherlock):
    out_dir = os.path.join(os.path.dirname(data_file), 'walk')
    validation_path = os.path.join(out_dir, "validation.hdf5")
    test_path = os.path.join(out_dir, "test.hdf5")
    
    if(os.path.exists(validation_path) and os.path.exists(test_path)):
        print("Loading existing data files")
        walk_train = test_path
        walk_validation = validation_path
            
    else:
        (walk_train, walk_validation, num_walk) = split_data(data_file, validation_split, 'walk')

#Make weights to balance the training set
#class_weights = {0: num_rest/(num_rest + num_walk), 1: num_walk/(num_rest + num_walk)}

#Alternatively, discard data so we have balanced training and validation sets
training_batch_generator = SixMWTSequence(walk_train, rest_train, batch_size, True)
validation_batch_generator = SixMWTSequence(walk_validation, rest_validation, batch_size, True)

num_training_samples = len(training_batch_generator)
num_validation_samples = len(validation_batch_generator)
num_epochs = 1000

#%%

#Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

early_stop = EarlyStopping(patience=7)

tb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))

history = model.fit_generator(generator=training_batch_generator,
                              epochs=num_epochs,
                              verbose=1,
                              callbacks = [reduce_lr, early_stop, tb],
                              validation_data=validation_batch_generator,
                              #class_weight=class_weights,
                              use_multiprocessing=canMultiprocess, #Windows must be False, else true
                              workers=8,
                              max_queue_size=32)

#Clean up the temp files
del training_batch_generator
del validation_batch_generator
#os.remove(walk_train)
#os.remove(rest_train)
#os.remove(walk_validation)
#os.remove(rest_validation)

#Save history and model
with open(os.path.join(output_dir, 'train_history.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
model.save(os.path.join(output_dir, "model.h5"))
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
    implement class weights
    play around with model architecture
    Speed up training - overhead is load/strs
    hyperparameter tuning

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
    data_file = "/scratch/PI/euan/projects/mhc/code/daniel_code/filtered_windows/data_windows_walk.hdf5"
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
    from the data_file, aggregates and labels healthcode windows, 
    and saves to a new hdf5 file with datasets representing a single window.
    '''
    
    #Complete file
    data_file = h5py.File(data_path, 'r')
    
    #The new file
    new_file = h5py.File(out_path, 'a')
    
    #Go through the keys of the old file 
    valid_codes = set(healthCodes)
    counter = 0
    for code in data_file.keys():
        #If the healthcode is valid, copy into new file
        if code in valid_codes:
            #Old organization - same heirarchy
            #data_file.copy(code, new_file)
            
            #Put each window as its own dataset in HDF
            for i in range(data_file[code].shape[0]):
            
                d = new_file.create_dataset(str(counter), data = data_file[code][i])
                d.attrs["healthCode"] = code
                counter += 1
            
    data_file.close()
    
    #Return the HDF file
    return new_file

def extract_data(data_file, out_file_name):
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
    out_path = os.path.dirname(data_file) + os.sep + out_file_name
    #Extract the records out from the 
    filtered_data = extract_records(healthCodes, data_file, out_path)
    
    return filtered_data, label_df

# =============================================================================
# Split data files into validation and test
# =============================================================================

def split_data(file_path, split):
    '''
    split_data(file_path, split)
    splits the data in the hdf file at filepath
    with the given split ratio, between 0 and 1.
    Uses new_folder in the new filepath
    Returns a tuple with two filenames, the first with (1-split), the second with (split) percent of the data
    '''
    with h5py.File(file_path, 'r') as hdf_file:
        
        health_codes = list(hdf_file.keys())
        #Make the split
        validation_codes = health_codes[: math.floor(len(health_codes) * split)]
        test_codes = health_codes[math.floor(len(health_codes) * split) :]
        
        #Define path of valiation file
        out_dir = os.path.dirname(file_path)
        validation_path = os.path.join(out_dir, "validation_unfiltered.hdf5")
     
        #Open and save the valdation set
        with h5py.File(validation_path, 'w') as validation_file:
            
            for code in validation_codes:
                hdf_file.copy(code, validation_file)
                
        #Open and save the test set
        test_path = os.path.join(out_dir, "test_unfiltered.hdf5")
        
        with h5py.File(test_path, 'w') as test_file:

            for code in test_codes:
                hdf_file.copy(code, test_file)
        
    return test_path, validation_path
        
# =============================================================================
# Data generator
# =============================================================================
def parse_label(code, label_df):
    """
    Helper function that parses the labels on survey data for a given code
    """
    if labels_of_interest == ["heartCondition"]:
        text_label = label_df.loc[code]
        
        #textlabel for heartCondition is a True False boolean
        if text_label.bool():
            return 1
        else:
            return 0
        
        
    

class SixMWTSequence(keras.utils.Sequence):
    '''
    SixMWTSequence
    Extends keras inbuilt sequence to create a data generator
    Saves on RAM by loading data from hdf5 files in memory
    __del__ way of closing files isn't great - find a better way sometime
    '''
    def __init__(self, data_file, batch_size, label_df):
        #Open up file
        self.lock = threading.Lock()
        self.file = data_file
        
        #Track labels and batch size
        self.labels = label_df
        self.batch_size = batch_size
        
        #Calculate length of points - len is too memory intensive
        self.num_data = 0
        for code in self.file.keys():
            self.num_data += 1
            
        #Partition the dataset into batches
        self.length = self.num_data // self.batch_size

    def __len__(self):
        #Find how many batches fit in our dataset
        #This "crops" out a couple datapoints not divisible by the batch at the end
        return self.length

    def __getitem__(self, idx):
        
        with self.lock:
            
            #Get the batch members
            batch_x = [self.file[str(i)][:] for i in range(idx*self.batch_size, (idx + 1)*self.batch_size)]
            batch_y = [parse_label(self.file[str(i)].attrs["healthCode"], self.labels) for i in range(idx*self.batch_size, (idx + 1)*self.batch_size)]
   
            #Convert to array
            batch_x = np.asarray(batch_x)
            batch_y = np.asarray(batch_y)
            
            return batch_x, batch_y
        
    def __del__(self):
        self.file.close()    

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
    out_dir = os.path.dirname(data_file)
    validation_path = os.path.join(out_dir, "validation_unfiltered.hdf5")
    test_path = os.path.join(out_dir, "test_unfiltered.hdf5")
    
    if(os.path.exists(validation_path) and os.path.exists(test_path)):
        print("Loading existing unfiltered data files")
        walk_train = test_path
        walk_validation = validation_path
            
    else:
        (walk_train, walk_validation) = split_data(data_file, validation_split)

#Make weights to balance the training set
#class_weights = {0: num_rest/(num_rest + num_walk), 1: num_walk/(num_rest + num_walk)}

#Filter the data, if not already filtered
out_path = os.path.dirname(walk_train) + os.sep + "filtered_train.hdf5"
if os.path.exists(out_path):
    print("Loading existing filtered train data")
    filtered_train_file = h5py.File(out_path, 'a')
    label_df = extract_labels(labels = labels_of_interest, label_table_path = label_table_file)
    
else:
    (filtered_train_file, label_df) = extract_data(walk_train, "filtered_train.hdf5")
    
training_batch_generator = SixMWTSequence(filtered_train_file, batch_size, label_df)

#Filter the data, if not already filtered
out_path = os.path.dirname(walk_validation) + os.sep + "filtered_validation.hdf5"
if os.path.exists(out_path):
    print("Loading existing filtered validation data")
    filtered_validation_file = h5py.File(out_path, 'a')
    label_df = extract_labels(labels = labels_of_interest, label_table_path = label_table_file)
    
else:
    (filtered_validation_file, label_df) = extract_data(walk_validation, "filtered_validation.hdf5")
    
validation_batch_generator = SixMWTSequence(filtered_validation_file, batch_size, label_df)

num_training_samples = len(training_batch_generator)
num_validation_samples = len(validation_batch_generator)
num_epochs = 1000


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
                              use_multiprocessing=canMultiprocess, 
                              workers=8,
                              max_queue_size=32)

#Clean up the temp files
del training_batch_generator
del validation_batch_generator

#Save history and model
with open(os.path.join(output_dir, 'train_history.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
model.save(os.path.join(output_dir, "model.h5"))
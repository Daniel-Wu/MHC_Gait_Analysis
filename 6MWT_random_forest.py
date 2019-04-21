# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:39:57 2019

Checking if there's a signal in the dataset
Uses random forest model

@author: dwubu
"""

from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import numpy as np

data_filename = r'C:\Users\dwubu\Desktop\6mwtInhouseFiltered\Walk'

label_dict = {'ben_walk_20099.h5' : 0, 
              'cameron_walk_20099.h5' : 0, 
              'chunli_walk_20099.h5' : 1, 
              'colleen_walk_20099.h5' : 1, 
              'hannah_walk_20099.h5' : 1, 
              'john_walk_20099.h5' : 0, 
              'malene_walk_20099.h5' : 1, 
              'marie_walk_20099.h5' : 1, 
              'megan_walk_20099.h5' : 1, 
              'oana_walk_20099.h5' : 1, 
              'yong_walk_20099.h5' : 1}


#Balanced Training sets
training_names = ['ben_walk_20099.h5', 'chunli_walk_20099.h5', 'hannah_walk_20099.h5','john_walk_20099.h5']
testing_names = ['cameron_walk_20099.h5', 'colleen_walk_20099.h5']

#Full Training set
#training_names = ['ben_walk_20099.h5', 'chunli_walk_20099.h5', 'hannah_walk_20099.h5','john_walk_20099.h5', 'malene_walk_20099.h5', 'marie_walk_20099.h5', 'megan_walk_20099.h5']
#testing_names = ['cameron_walk_20099.h5', 'colleen_walk_20099.h5', 'oana_walk_20099.h5', 'yong_walk_20099.h5']

#Import in the data
def import_data(filepath, label_dict, train_names, test_names):

    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for filename in os.listdir(filepath):
        
        data = pd.read_hdf(os.path.join(filepath, filename))
            
        for i in range(len(data)):
            
            new_x = [data.iloc[i]['xwindows'], data.iloc[i]['ywindows'], data.iloc[i]['zwindows']]
            
            if filename in train_names:
                X_train.append(new_x)
                y_train.append(label_dict[filename])
            elif filename in test_names:
                X_test.append(new_x)
                y_test.append(label_dict[filename])
            else:
                print("UNKNOWN FILE NAME {}".format(filename))
            
    
            
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        
        
        
X_train, y_train, X_test, y_test = import_data(data_filename, label_dict, training_names, testing_names)



X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))


#Shuffle X and y
if(True):
    shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_idx)
    
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
        
    shuffle_idx = np.arange(X_test.shape[0])
    np.random.shuffle(shuffle_idx)
    
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]
    

#Initialize the classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features=0.3)#, max_depth = 5, max_features=0.3)

#Fit the classifier
rf.fit(X_train, y_train)


if(False):
    
    #Crossvalidation
    depths = [3,5,8,10]
    features = [0.1,0.3,0.5,0.8]
    
    num_folds = 5
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    accuracies = {}
    
    
    for depth in depths:
        for feature in features:
    
            accs = []
    
            for i in range(num_folds):
                #Concatenate together the data folds
                fold_idxs = [idx for idx in range(num_folds) if idx != i]
                fold_data = np.concatenate([X_train_folds[fold_idx] for fold_idx in fold_idxs])
                fold_Y = np.concatenate([y_train_folds[fold_idx] for fold_idx in fold_idxs])
        
                rf = RandomForestClassifier(n_estimators=10, max_depth = depth, max_features=feature)
                rf.fit(fold_data, fold_Y)
                fold_pred = rf.predict(X_test)
                
                #Calculate our accuracies
                new_acc = np.sum(fold_pred == y_test)/len(fold_pred)
                accs.append(new_acc)
            
            #Store accuracies in our dictionary    
            accuracies[(depth,feature)] = accs
    
    
    # Print out the computed accuracies
    for k in sorted(accuracies):
        for accuracy in accuracies[k]:
            print('params = {}, accuracy = {}'.format(k, accuracy))
            

#Assess Accuracy
train_pred = rf.predict(X_train)
train_acc = np.sum(train_pred == y_train)/len(train_pred)
print("Training accuracy: {}".format(train_acc))

test_pred = rf.predict(X_test)
test_acc = np.sum(test_pred == y_test)/len(test_pred)
print("Test accuracy: {}".format(test_acc))

#Results on internal toy dataset
#Training accuracy: 0.9958419958419958
#Test accuracy: 0.7330567081604425
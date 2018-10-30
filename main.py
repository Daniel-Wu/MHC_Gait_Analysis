# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:48:59 2018

This script makes an autoencoder with keras
Denoises gait data

@author: dwubu
"""

from keras.layers import Input, Dense
from keras.models import Sequential

from keras.datasets import mnist
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

#Load in the json data
with open(r"C:\Users\dwubu\Desktop\6mwtData\2687897.json") as json_data:
    walk_json = json.load(json_data)

#convert json entries into better vectors
walk_test_data = {'x' : [],
                  'y' : [],
                  'z' : [],
                  'timestamp' : []}

for data_point in walk_json:
    walk_test_data['x'].append(data_point['x'])
    walk_test_data['y'].append(data_point['y'])
    walk_test_data['z'].append(data_point['z'])
    walk_test_data['timestamp'].append(data_point['timestamp'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = walk_test_data['x']
y = walk_test_data['y']
z = walk_test_data['z']
ax.plot(x, y, z)
plt.show()

if(True):
    input_dim = len(x)
    encoding_dim = 64
    
    print(len(x))
    
    ### The sequential model
    autoencoder = Sequential()
    autoencoder.add(Dense(4*encoding_dim, input_shape=(input_dim,),activation='relu'))
    autoencoder.add(Dense(2*encoding_dim,activation='relu'))
    autoencoder.add(Dense(encoding_dim,activation='relu'))
    autoencoder.add(Dense(2*encoding_dim,activation='relu'))
    autoencoder.add(Dense(4*encoding_dim,activation='relu'))
    autoencoder.add(Dense(input_dim,activation='sigmoid'))
    
    # configuring stuff
    autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
    
    
    # get mnist data, discarding labels.
    #(x_train, _), (x_test, _) = mnist.load_data()
    #Normalize between 0 and 1
    #x_train = x_train.astype('float32') / 255
    #x_test = x_test.astype('float32') / 255
    
    #Reshape images to vectors
    x = np.true_divide(x, max(x))
    x = np.array([x, x])

    x_train = x;
    x_test = x;
    
#    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#    print(x_train.shape)
#    print(x_test.shape)
    
    
    #TRAINING
    autoencoder.fit(x_train, x_train,
                    epochs=500,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    #try some images
    #decoded_imgs = autoencoder.predict(x_test)
    
    #n = 20  # how many digits we will display
    #plt.figure(figsize=(20, 4))
    #for i in range(n):
#        # display original
#        ax = plt.subplot(2, n, i + 1)
#        plt.imshow(x_test[i].reshape(28, 28))
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#    
#        # display reconstruction
#        ax = plt.subplot(2, n, i + 1 + n)
#        plt.imshow(decoded_imgs[i].reshape(28, 28))
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#    plt.show()
#    
#    autoencoder.summary()

    plt.plot(x_train)
    plt.plot(autoencoder.predict(x_train))
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:10:34 2018

Extracts valid windows from 6mwt data

@author: dwubu
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def validate_window(series):
    '''
    validate_window
    Checks a pandas dataframe to see if the data is anomalous
    '''
    return True

def hann(window):
    '''
    hann
    Applies a hanning window function to the passed vector
    '''
    return np.hanning(len(window))*window

def ihann(window):
    '''
    hann
    Inverts the hanning window function to the passed vector
    '''
    return window/np.hanning(len(window))


# =============================================================================
# Functions for applying the bandpass filter
# =============================================================================

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# =============================================================================
# Begin Main
# =============================================================================

window_directory_str = 'C:/Users/dwubu/Desktop/6mwtWindows'
window_directory = os.fsencode(window_directory_str)


for filename in os.listdir(window_directory):
    
    #Read in the hdf of each file
    filepath = os.path.join(window_directory.decode(), filename.decode())
    data = pd.read_hdf(filepath)
    

#Indent later to do more than one data file    
for index, series in data.iterrows():
    #For each window
    pass


#Indent later to do more than one series
    #print(series)
print(index)
print("=====================================================================")
#Print original data    
plt.figure()
plt.title("Original Data")
plt.plot(series['xwindows'], label = 'x')
plt.plot(series['ywindows'], label = 'y')
plt.plot(series['zwindows'], label = 'z') 
plt.legend()   
plt.show()


#Extract data
x = series['xwindows']
y = series['ywindows']
z = series['zwindows']

#Apply butterfield window 
lowcut = 15
highcut = 40
fs = 100

x = butter_bandpass_filter(x, lowcut, highcut, fs)
y = butter_bandpass_filter(y, lowcut, highcut, fs)
z = butter_bandpass_filter(z, lowcut, highcut, fs)


plt.figure()
plt.title("Butterfield Filtered")
plt.plot(abs(x[1:]), label = 'x')
plt.plot(abs(y[1:]), label = 'y')
plt.plot(abs(z[1:]), label = 'z')
plt.legend()
plt.show()

    #Frequency transformed data, hann windowed
x = np.fft.fft(hann(x))
y = np.fft.fft(hann(y))
z = np.fft.fft(hann(z))

plt.figure()
plt.title("FFT Transform")
plt.plot(abs(x[1:]), label = 'x')
plt.plot(abs(y[1:]), label = 'y')
plt.plot(abs(z[1:]), label = 'z')
plt.legend()
plt.show()



#plt.figure()
#plt.title("Inverse Transform")
#plt.plot(abs(np.fft.ifft(x)))
#plt.plot(abs(np.fft.ifft(y)))
#plt.plot(abs(np.fft.ifft(z)))
#plt.show()

#final_data = pd.Dataframe

# =============================================================================
# Saving to a file
# =============================================================================
if(False):
    final_directory_str = window_directory_str + "/Processed"
    final_directory = os.fsencode(window_directory)
    if not os.path.exists(final_directory.decode()):          
        os.makedirs(final_directory.decode())
    
    final_data.to_hdf(final_directory.decode() + '/' + filename +'.h5', key='df', mode='w')

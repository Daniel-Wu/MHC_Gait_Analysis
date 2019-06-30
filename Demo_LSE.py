# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 06:55:48 2019
Test LSE
@author: dwubu
"""

import pandas as pd
import numpy as np
import lomb_scargle_extractor as lse
import matplotlib.pyplot as plt

filename = r'yong_walk.json'
df = pd.read_json(filename)

#Conform to api - reindex, add mag column, and rename
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
temp_idx = pd.DatetimeIndex(df['timestamp'])
df = df.drop('timestamp', axis = 1)
norm = np.sqrt(np.square(df).sum(axis=1))
df = pd.concat([df, norm], axis=1)
df = df.rename(index=str, columns={'x': 'a_x', 'y': 'a_y', 'z': 'a_z', 0: 'a_mag'})
df = df.set_index(temp_idx)

#Extract lse
data = df[2000:2400]
test_cycle = lse.extract(data, plot_vis=True)
if(isinstance(test_cycle, pd.DataFrame)):
    print("Plotting LSE Extraction")
    test_cycle = np.reshape(list(test_cycle['a_mag']), (100, -1), order='F')
    
    plt.figure()
    plt.plot(test_cycle)
    plt.title('Demo of LSE on Yong\'s walk data')
    plt.xlabel('Time')
    plt.ylabel('Absolute acceleration (g)')
    plt.savefig('lse_demo.png')
else:
    print("No windows found")
    

#Make full windows and plot summary
chunk_size = 400
cycles = np.empty((0, 100))
for idx in range(0, df.shape[0], chunk_size):

    data = df[idx: idx+chunk_size]
    
    try:
        test_cycle = lse.extract(data)
    except:
        continue
    
    if(isinstance(test_cycle, pd.DataFrame)):
        test_cycle = np.reshape(list(test_cycle['a_mag']), (100, -1), order='F').T
        cycles = np.vstack((cycles, test_cycle))

#Plot the cycles
plt.figure(2)
for row in cycles:
    plt.plot(row, 'b', alpha = 0.05)
    
plt.title('Average walk cycle - Yong')
plt.ylabel('Magnitude (g)')
mean_cycle = np.mean(cycles, axis = 0)
plt.plot(mean_cycle, 'r')
plt.savefig('yong_walk_cycle.png')
plt.show()
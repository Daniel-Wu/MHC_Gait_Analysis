# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:11:54 2019

Extracts aws activity reports for pretty graph making

@author: Daniel Wu
"""

import gzip
import json

important_events = ['pageStarted']
out_file_path = '/scratch/PI/euan/projects/mhc/code/daniel_code/aws'
out_file = open(out_file_path, 'a')


def criteria(json_data):
    '''
    Helper function that returns a bool representing whether to process json_data
    '''
    return datapoint['event_type'] in important_events

def process_data(json_data):
    '''
    Helper function that processes a single json_data aws record
    '''
    pass

# =============================================================================
# Do the actual data parsing
# =============================================================================

#Load file
with open('/scratch/PI/euan/projects/mhc/data/aws.all', 'r') as filenames:
    
    for f in filenames:
        #load and read in file data
        with gzip.open(f,'rb') as jf: 
            #Each .gz contains multiple jsons and some random empty (unix) linebreaks
            data=[json.loads(line) for line in jf.read().decode('UTF-8').split('\n') if line!='']
            
            #Iterate through each record in the file
            for datapoint in data:
                #Process valid datapoints
                if(criteria(datapoint)):
                    process_data(datapoint)


#CLOSE THE FILE
out_file.close()
            
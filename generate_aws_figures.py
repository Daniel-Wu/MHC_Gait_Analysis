# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:25:04 2019

Generates various figures for AWS Demographics analytics

@author: Daniel Wu
"""

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt

#Anna's function to map AWS client IDs to healthcodes, returns dict
def map_aws_to_healthcode(source_table = r'/scratch/PI/euan/projects/mhc/code/daniel_code/aws/tables/cardiovascular-AwsClientIdTask-v1.tsv'): 
    client_id_to_health_code_id=dict()
    #read in the data 
    dtype_dict=dict() 
    dtype_dict['names']=('skip',
                         'recordId',
                         'appVersion',
                         'phoneInfo',
                         'uploadDate',
                         'healthCode',
                         'externalId',
                         'dataGroups',
                         'createdOn',
                         'createdOnTimeZone',
                         'userSharingScope',
                         'validationErrors',
                         'AwsClientId')
    dtype_dict['formats']=('S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36',
                           'S36')
    try: 
        data=np.genfromtxt(source_table,
                           names=dtype_dict['names'],
                           dtype=dtype_dict['formats'],
                           delimiter='\t',
                           skip_header=True,
                           loose=True,
                           invalid_raise=False)
    except:
        print("failed to load file:"+str(source_table))
        raise Exception() 

    #create a mapping of client id to healthCode 
    for line in data: 
        client_id_to_health_code_id[line['AwsClientId'].decode('UTF-8')] = line['healthCode'].decode('UTF-8')    
    return client_id_to_health_code_id


def reject_outliers(data, m=3):
    outliers = data[np.logical_or(abs(data - np.mean(data)) >= m * np.std(data), data <= 2)]
    clean = data[np.logical_and(abs(data - np.mean(data)) < m * np.std(data), data>2)]
    
    num_less = (data <= 2).sum()#(outliers < np.mean(data)).sum()
    num_more = (outliers > np.mean(data)).sum()

    return clean, num_less, num_more

def page_duration():
    '''generates a table and figure describing average time spent on an activity based on activity'''
    
    print("Generating page duration table")
    
    freq = defaultdict(list)
    
    with open(r'/scratch/PI/euan/projects/mhc/code/daniel_code/aws/tables/PageEndedActivities.txt', 'r') as file:
        print("Loading in data")
        for line in file:
            data = json.loads(line.rstrip())
            dur = int(data["attributes"]["duration"])
            act = data["attributes"]["pageName"]
            
            freq[act].append(dur)
            
    print("Finished loading the data - beginning parsing")
    #Describe all the results
    for key in freq.keys():
        print("Stats for activity {}".format(key))
        print(stats.describe(freq[key]))
        
        #clean up the data a little
        clean, num_less, num_more = reject_outliers(np.array(freq[key]))
        
        #Plot everything
        fig = plt.hist(clean)
        plt.title('Duration spent on activity {} \n {} 2 or less, {} high outliers'.format(key, num_less, num_more))
        plt.xlabel("Duration")
        plt.ylabel("# Occurences")
        plt.savefig("/scratch/PI/euan/projects/mhc/code/daniel_code/aws/figures/duration_{}.png".format(key))
        plt.clf()

def age_activity(healthcode_map):

    '''Generates a figure and table of the average duration spent on various activities depending on age'''
    
    print("Generating age activity table")

    
    age_table = pd.read_csv(r'tables/demographics_summary_v2.age.tsv', delimiter='\t').set_index('Subject')
    
    freq = defaultdict(list)
    
    with open(r'/scratch/PI/euan/projects/mhc/code/daniel_code/aws/tables/PageEndedActivities.txt', 'r') as file:
        print("Loading in data")
        for line in file:
            data = json.loads(line.rstrip())
            aws_id = data["client"]["client_id"]
            act = data["attributes"]["pageName"]
            
            #Get the age
            try:
                healthcode = healthcode_map[aws_id]
                age = int(age_table.loc[healthcode, 'Agex'])
                
                freq[act].append(age)
            except KeyError:
                print("AWS_id not found {}".format(aws_id))
            
    print("Finished loading the data - beginning parsing")
    #Describe all the results
    for key in freq.keys():
        print("Stats for activity {}".format(key))
        print(stats.describe(freq[key]))
        
        
        #Plot everything
        fig = plt.hist(freq[key])
        plt.title('Age spent on activity {}'.format(key))
        plt.xlabel("Age")
        plt.ylabel("# Occurences")
        plt.savefig("/scratch/PI/euan/projects/mhc/code/daniel_code/aws/figures/age_{}.png".format(key))
        plt.clf()

if __name__ == '__main__':
    
    healthcode_map = map_aws_to_healthcode()
    age_activity(healthcode_map)
    
    #page_duration()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:02:48 2015

@author: USF
"""
# standard imports
import copy
import time
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

def extractCSV(file_path, target, id_column = 'id_list'):
    """
    Takes in a file path to a given CSV,
    a target (aka what we want to label the data),
    and the name of the id column
    
    Returns:
        1) x = numpy array, with id's removed
        2) y = list of either 0's or 1's
        3) ids = list of id's corresponding to the observations in the x-matrix
    """
    # read in the data
    data = pd.read_csv(file_path, header = 0)
    # remove the id column
    ids = data.pop(id_column).tolist()
    # create the x-matrix
    x = data.as_matrix()
    # and create the corresponding y target values
    y = np.asarray([target] * len(x))
    # now we can return
    return x, y, ids

def singleDriverTrainer2(file_to_classify, training_files, threshold = .3,
                        in_model = RandomForestClassifier()):
    """
    Takes in the file path of the driver file we want to classify (the target),
    the paths of the files we will use as our 'noise' files,
    and the input model
    
    First, trains the input model on all of the files, with file_to_classify
    as class 1 and training_files as class 0
    
    Then, uses the model to make probabilistic predictions on file_to_classify
    
    Changes: Using the probabilistic predictions
    relabels 1s to 0s based on threshhold
    """
    # first, grab the target data
    x_target, y_target, id_target = extractCSV(file_to_classify, target = 1)

    # remove na's
    x_target = np.nan_to_num(x_target)
    y_target = np.nan_to_num(y_target)
    
    # now grab the training/noise data
    x_all = copy.copy(x_target)
    y_all = copy.copy(y_target)
    
    n = len(training_files)
    l = len(x_target)
    
    #upsample target to balance classes
#    if n > 1:
#        upsample_idx = np.random.choice(range(l), l*n)
#        x_all = x_all[upsample_idx]
#        y_all = y_all[upsample_idx]
    
    #stack target to balance classes    
    if n > 1:
        stack_idx = range(l) * n
        x_all = x_all[stack_idx]
        y_all = y_all[stack_idx]
        

    # loop through all of our training/noise files, keep separate from target
    for filepath in training_files:
        # open the file
        x_current, y_current, ids = extractCSV(filepath, target = 0)
        try:
            # and add the contents to our training data
            x_trains = np.concatenate((x_trains, x_current))
            y_trains = np.concatenate((y_trains, y_current))
        except:
            x_trains, y_trains = x_current, y_current
    # repeat for every filepath in our training files list
        
    #remove NAs from train data
    x_trains = np.nan_to_num(x_trains)
    y_trains = np.nan_to_num(y_trains)
    
    #now combine with target data
    x_all = np.concatenate((x_all, x_trains))
    y_all = np.concatenate((y_all, y_trains))
        
    # with all of our data, now we can train our model
    in_model.fit(x_all, y_all)
    
    # now we are ready to provide class probabilities for our predictions
    predictions = in_model.predict_proba(x_target)
    
    # note that we must extract the index of the class 1 probability
    prob_idx = np.where(in_model.classes_ == 1)[0][0]
    class_probs = [pred[prob_idx] for pred in predictions]
    
    #get new data labels by comparing threshold to class probs
    new_labels = np.array([1 if p > threshold else 0 for p in class_probs])
    #redo upsampling
    upsample_idx = np.random.choice(range(l), l*n)
    x_all = x_target[upsample_idx]
    y_all_new = new_labels[upsample_idx]
    #or redo stacking
    if n > 1:
        x_all = x_target[stack_idx]
        y_all_new = new_labels[stack_idx]
    
    #combine with non-target data from before
    x_all = np.concatenate((x_all, x_trains))
    y_all_new = np.concatenate((y_all_new, y_trains))
    
    #refit model
    in_model.fit(x_all, y_all_new)
    # now we are ready to provide class probabilities for our predictions
    predictions = in_model.predict_proba(x_target)
    # note that we must extract the index of the class 1 probability
    prob_idx = np.where(in_model.classes_ == 1)[0][0]
    class_probs = [pred[prob_idx] for pred in predictions]
    # and return a matrix of the id's and the corresponding probabilities
    return_mat = [[id_target[idx], class_probs[idx]] \
                    for idx in xrange(len(class_probs))]
    
    return np.asarray(return_mat)
    
    
classify_file = '1_summary.csv'
train_files = ['2_summary.csv','3_summary.csv','10_summary.csv','11_summary.csv']

train_file_count = len(train_files)
num_features = min(28, (train_file_count+1)*4)

model = BaggingClassifier(LogisticRegression(), n_estimators = 100,
                          max_features = num_features)
                          
results = singleDriverTrainer2(classify_file, train_files, in_model = model)


__author__ = 'mshadish'
"""
The general idea is to build a Random Forest to classify drivers
as to whether or not they belong in a given folder

1) Take all of the files in a given folder, label them as class 1
2) Take files from (5) other folders, label them as class 0
3) Train a random forest on this pseudo-labeled data
4) Predict on the files from the given folder in Step 1
    - Use the assigned probabilities to say, with some probability,
    whether or not that driving record belongs in the given folder
    
The hope is that, for a given folder, some defining feature of the driver's
fingerprint will stand out, and all of the "noise" will fall away.
"""
# standard imports
import os
import copy
import time
import random
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.ensemble import RandomForestClassifier


def genListOfCSVs(path):
    """
    Takes in a given path
    
    Returns a list of all of the CSV's in the given path
    """
    # grab a list of everything in this path
    all_files = os.listdir(path)
    # extract out the CSV's
    csvs = [i for i in all_files if i.split('.')[-1].lower() == 'csv']
    return csvs
    
    
    
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
    
    
    
def genTrainingSet(set_of_CSVs, file_to_classify, train_size = 5):
    """
    Takes in a list of CSV file paths
    as well as the file path of the CSV we want to 'classify'
    AKA pull out the 'true' driver data from the noise
    
    Returns the names of 5 file paths chosen at random
    that will serve as our 'training' set
    AKA will be the random noise we will classify as 0
    """
    set_of_csvs_minus_target = copy.copy(set_of_CSVs)
    # remove the file we want to classify
    set_of_csvs_minus_target.remove(file_to_classify)
    
    # extract out the random noise files
    # first, set the seed
    random.seed(time.time())
    # now sample
    return_list = random.sample(set_of_csvs_minus_target, train_size)
    return return_list
    
    
    
def singleDriverWrapper(file_to_classify, training_files,
                        in_model = RandomForestClassifier()):
    """
    Takes in the file path of the driver file we want to classify (the target),
    the paths of the files we will use as our 'noise' files,
    and the input model
    
    First, trains the input model on all of the files, with file_to_classify
    as class 1 and training_files as class 0
    
    Then, uses the model to make probabilistic predictions on file_to_classify
    """
    # first, grab the target data
    x_target, y_target, id_target = extractCSV(file_to_classify, target = 1)
    # remove na's
    x_target = np.nan_to_num(x_target)
    y_target = np.nan_to_num(y_target)
    
    # now grab the training/noise data
    x_all = copy.copy(x_target)
    y_all = copy.copy(y_target)
    # loop through all of our training/noise files
    for filepath in training_files:
        # open the file
        x_current, y_current, ids = extractCSV(filepath, target = 0)
        # and add the contents to our training data
        x_all = np.concatenate((x_all, x_current))
        y_all = np.concatenate((y_all, y_current))
    # repeat for every filepath in our training files list
        
    # again, remove na's from our data
    x_all = np.nan_to_num(x_all)
    y_all = np.nan_to_num(y_all)
        
    # with all of our data, now we can train our model
    in_model.fit(x_all, y_all)
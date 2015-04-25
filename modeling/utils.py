__author__ = 'mshadish'
"""
Utility functions built for generalizability

1. genListOfCSVs(path)
    - takes in a path
    - returns a list of all the full paths to files in the given path
    that have a .csv extension
        
2. bootstrap(x, y, sample size)
    - bootstraps x and y to a specified sample size
    - returns bootstrapped x and y

3. stackUpsample(x, y, multiple size)
    - similar to our bootstrap function,
    but instead will "bootstrap" by increasing the number of occurrences
    of each data point by our given multiple

4. edit_probs(name of submission file, lower threshold, upper threshold)
    - reads in a given submission file
    - pushes any probabilities below our lower threshold to 0
    and any probabilities above our upper threshold to 1
    - ideally, this could increase our AUC if we are fairly confident
    about the predictions outside of the lower and upper thresholds
"""
# imports
import os
import re
import pandas as pd
import numpy as np


def genListOfCSVs(path):
    """
    Takes in a given path
    
    Returns a list of all of the CSV's in the given path
    """
    # grab a list of everything in this path
    all_files = os.listdir(path)
    # extract out the CSV's
    csvs = [i for i in all_files if i.split('.')[-1].lower() == 'csv']
    # add the path back into the file path
    path = re.sub(r'\/$', '', path)
    csvs = ['/'.join([path, i]) for i in csvs]

    return csvs
    
    
    
def bootstrap(input_x, input_y, sample_size):
    """
    Bootstraps a sample of size 'sample_size'
    by sampling (with replacement) randomly from input_x and input_y
    
    Returns:
        1) output_x
        2) output_y
    """
    l = len(input_x)
    # generate our up-sample index
    upsample_idx = np.random.choice(range(l), sample_size)
    # with these indices, extract the corresponding observations
    output_x = input_x[upsample_idx]
    output_y = input_y[upsample_idx]

    return output_x, output_y
    
    
    
def stackUpsample(input_x, input_y, multiple):
    """
    Stacks a sample on top of itself to return an upsampled version
    of the original input data
    
    Returns:
        1) output_x, the upsampled x points
        2) output_y, the corresponding labels
    """
    output_x = np.asarray(input_x).tolist() * multiple
    output_y = np.asarray(input_y).tolist() * multiple
    return output_x, output_y



def edit_probs(file_name, low, high):
    data = np.asarray(pd.read_csv(file_name+'.csv', header=0))

    target = [0.0 if data[i, 1] <= low else data[i, 1] for i in range(len(data[:, 1]))]
    target = np.asarray([1.0 if target[i] >= high else target[i] for i in range(len(target))])
    edited = np.vstack((data[:, 0], target))

    df = pd.DataFrame(edited.transpose(), columns=['driver_trip', 'prob'])
    df.to_csv(file_name+'_edited.csv', index=False)

    return

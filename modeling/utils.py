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
"""
# imports
import os
import re
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
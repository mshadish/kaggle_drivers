__author__ = 'mshadish'
"""
Utility functions built for generalizability

1. genListOfCSVs()
    - takes in a path
    - returns a list of all the full paths to files in the given path
    that have a .csv extension
"""
# imports
import os
import re

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
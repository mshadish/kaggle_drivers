__author__ = 'monicameyer'

import pandas as pd
import os
import re
import csv

from collections import Counter
from scipy.stats.stats import pearsonr


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


def extractCSV(file_path, id_column='id_list'):
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
    data = pd.read_csv(file_path, header=0)
    # remove the id column
    ids = data.pop(id_column).tolist()
    # print data.columns.values
    # create the x-matrix
    x = data.as_matrix()

    c = []
    for i in range(len(x[0])):
        for j in range(len(x[0])):
            if i < j:
                p = abs(pearsonr(x[:, i], x[:, j])[0])
                if p > .7 and p < 1:
                    c.append([i, j])

    return x, ids, c, data.columns.values



if __name__ == '__main__':

    all_files = genListOfCSVs('../extracted')
    correlations = Counter()
    for file in all_files:
        print file
        x_target, id_target, corr, names = extractCSV(file)
        for i in corr:
            correlations[names[i[0]] + ', ' + names[i[1]]] += 1

    ordered_corr = correlations.most_common()

    header = ['variables', 'files_correlated']
    writer = csv.writer(open('correlation.csv', 'wb'))
    writer.writerow(header)
    for row in ordered_corr:
        writer.writerow([row[0], row[1]])

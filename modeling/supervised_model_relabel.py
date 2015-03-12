__author__ = 'mshadish'
"""
Note that we can define the global constants below for ease of use

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
import copy
import time
import random
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
# utility imports
from utils import genListOfCSVs
from utils import bootstrap
from utils import stackUpsample
# import for parallelization
from multiprocessing import Pool


# let's define our global constants here
# for ease of modification
# path of the summary files
path = '/users/mshadish/git_repos/kaggle_drivers/extracted'
# path for output file name
output_filename = 'solutions_4files_20feats_relabeled_01thresh.csv'
all_files = genListOfCSVs(path)
# number of training/noise files to use
train_file_count = 4
# specify the number of features, for simplicity
num_features = min(32, (train_file_count+1)*4)
# model to use
model = BaggingClassifier(LogisticRegression(), n_estimators = 100,
                          max_features = num_features)

    
    
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
    
    
    
def singleDriverTrainer2(file_to_classify, training_files, threshold = 0.1,
                        in_model = RandomForestClassifier()):
    """
    Takes in the file path of the driver file we want to classify (the target),
    the paths of the files we will use as our 'noise' files,
    and the input model
    
    First, trains the input model on all of the files, with file_to_classify
    as class 1 and training_files as class 0
    
    Then, uses the model to make probabilistic predictions on file_to_classify
    
    Changes: 
    1. Upsamples target data to balance classes for model training
    2. Uses probabilistic predictions relabels 1s to 0s based on threshhold
    """
    # first, grab the target data
    x_target, y_target, id_target = extractCSV(file_to_classify, target = 1)

    # remove na's
    x_target = np.nan_to_num(x_target)
    y_target = np.nan_to_num(y_target)
    
    # copy target data
    x_target_upsampled = copy.copy(x_target)
    y_target_upsampled = copy.copy(y_target)
                                                           
    #stack target to balance classes test   
    if n > 1:
        stack_idx = range(l) * n
        x_all, y_all = x_all[stack_idx], y_all[stack_idx] 
        
    x_trains = None
    y_trains = None

    # loop through all of our training/noise files, keep separate from target
    for filepath in training_files:
        # open the file
        x_current, y_current, ids = extractCSV(filepath, target = 0)
        # and add the contents to our training data
        if x_trains is None or y_trains is None:
            x_trains = x_current
            y_trains = y_current
        else:
            x_trains = np.concatenate((x_trains, x_current))
            y_trains = np.concatenate((y_trains, y_current))
    # repeat for every filepath in our training files list
        
    # remove NAs from train data
    x_trains = np.nan_to_num(x_trains)
    y_trains = np.nan_to_num(y_trains)
    
    # now combine with target data
    x_all = np.concatenate((x_target_upsampled, x_trains))
    y_all = np.concatenate((y_target_upsampled, y_trains))
        
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
    if len(training_files) > 1:
        num_samples = len(x_target) * len(training_files)
        x_target_relabeled, y_target_relabeled = bootstrap(x_target,
                                                           new_labels,
                                                           num_samples)
    else:
        x_target_relabeled = copy.copy(x_target)
        y_target_relabeled = copy.copy(new_labels)

    
    #combine with non-target data from before
    x_all_new = np.concatenate((x_target_relabeled, x_trains))
    y_all_new = np.concatenate((y_target_relabeled, y_trains))
    
    # refit model
    in_model.fit(x_all_new, y_all_new)
    # provide class probabilities for our predictions
    predictions = in_model.predict_proba(x_target)
    # extract the index of the class 1 probability
    prob_idx = np.where(in_model.classes_ == 1)[0][0]
    class_probs = [pred[prob_idx] for pred in predictions]
    
    # and return a matrix of the id's and the corresponding probabilities
    return_mat = [[id_target[idx], class_probs[idx]] \
                    for idx in xrange(len(class_probs))]
    # report
    print 'completed driver %s' % file_to_classify
    
    return np.asarray(return_mat)
    
    
def parallelWrapper(target_file):
    """
    This function serves as a wrapper for parallelization of classification
    of each individual target file
    """
    # for readability, specifying the globals used
    global model
    global all_files
    global train_file_count
    # open the training files
    train_file_names = genTrainingSet(all_files, target_file,
                                      train_size = train_file_count)
    # return the results of running our single driver through the model
    return singleDriverTrainer2(target_file, train_file_names, in_model = model)
    
    
    
if __name__ == '__main__':
    # initialize the pool for parallelization
    p = Pool()
    # and parallelize prediction for each file
    pred_arrays = p.map(parallelWrapper, all_files)
    # now reduce these into a single result
    predictions_combined = reduce(lambda a,b: np.vstack((a,b)), pred_arrays)
    # and write to a csv
    df = pd.DataFrame(predictions_combined, columns = ['driver_trip', 'prob'])
    df.to_csv(output_filename, index = False)
    pass
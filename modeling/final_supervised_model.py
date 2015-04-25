__author__ = 'mshadish'
"""
Note that we define the global constants below our imports for ease of use

The idea is to build two models, a bagged logistic regression ensemble
and a random forest, and use these models to make probabilistic predictions
on each driving trip

Methodology:
1) Take all of the trips/files in a given folder, label them as class 1
2) Take data from (5) other folders, label them as noise class 0
3) Bootstrap data from our target class 1 to match the number of observations
    from class 0
4) Train a model on this pseudo-labeled data
5) Predict on the files from the given folder in Step 1
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
# import for parallelization
from multiprocessing import Pool


# let's define our global constants here for ease of modification

# path of the summary files
path = '/users/mshadish/git_repos/kaggle_drivers/extracted'
# use this path to create a list of all of our data files
all_files = genListOfCSVs(path)

# path for output solutions file
output_filename = 'solutions.csv'

# amount of training/noise data to use
train_file_count = 280
num_samples_per_training_file = 10

# specify the number of features bootstrapped by each ensemble
num_features_model1 = 12
num_features_model2 = 25

# models to use
model1 = RandomForestClassifier(n_estimators = 200,
                                max_features = num_features_model1,
                                max_depth = 3)
model2 = BaggingClassifier(LogisticRegression(), n_estimators = 50,
                           max_features = num_features_model2)



def extractCSV(file_path, target, id_column = 'id_list', file_subset = None):
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
    ids = data.pop(id_column)
    # create the x-matrix
    x = data.as_matrix()
    # and create the corresponding y target values
    y = np.asarray([target] * len(x))
    
    # if our file subset variable is not None, then we will take
    # a random subset of the x
    if file_subset is not None:
        # create some random indices
        random_indices = random.sample(range(len(x)), file_subset)
        # sample x
        x = x[random_indices]
        # and grab the corresponding id's
        ids = ids[random_indices]
        # cut down y to match the length of x
        y = y[:file_subset]
        
    # now we can return
    return x, y, ids.tolist()



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



def singleDriverTrainer(file_to_classify, training_files, in_model1, in_model2,
                        weight_1 = 0.5, file_subset_size = 200):
    """
    This function trains a model and generates predictions
    for a SINGLE driver file.  This function is then mapped to all of our files
    
    First, trains the input models on all of the files, with file_to_classify
    as class 1 and training_files as class 0.  Then, uses the model to make
    probabilistic predictions on file_to_classify
    
    :param file_to_classify = name of driver file of interest
    :param training_files = list of file names from which we will draw our
                            noise data
    :param in_model1 = first model that will be used
    :param in_model2 = second model
    :param weight_1 = weighting to be applied to the predictions of model 1
    :param file_subset_size = specifies how many observations to pull out
                                of each noise driver file
                                
    Returns:
        A numpy array of [trip id, probability]
    """
    # first, grab the target data
    x_target, y_target, id_target = extractCSV(file_to_classify, target = 1)
    # remove na's
    x_target = np.nan_to_num(x_target)
    y_target = np.nan_to_num(y_target)

    # now grab the training/noise data
    x_all = copy.copy(x_target)
    y_all = copy.copy(y_target)
    
    # use stacking to up-sample our target data
    # and balance it with our noise data
    n = int(round(len(training_files) * file_subset_size / 200.0))
    l = len(x_target)
    # stack target to balance classes test   
    if n > 1:
        stack_idx = range(l) * n
        x_all, y_all = x_all[stack_idx], y_all[stack_idx]

    # now we must add the data from each noise file
    # loop through all of our training/noise files
    for filepath in training_files:
        # open the file
        x_current, y_current, ids = extractCSV(filepath, target = 0,
                                               file_subset = file_subset_size)
        # and add the contents to our training data
        x_all = np.concatenate((x_all, x_current))
        y_all = np.concatenate((y_all, y_current))
    # repeat for every filepath in our training files list

    # again, remove na's from our data
    x_all = np.nan_to_num(x_all)
    y_all = np.nan_to_num(y_all)


    ###########
    # MODEL 1 TRAINING
    ###########
    in_model1.fit(x_all, y_all)
    # now we are ready to provide class probabilities for our predictions
    # on the target data
    predictions = in_model1.predict_proba(x_target)
    # note that we must extract the index of the class 1 probability
    prob_idx = np.where(in_model1.classes_ == 1)[0][0]
    # apply the weighting for model 1 to each prediction
    class_probs1 = [pred[prob_idx]*weight_1 for pred in predictions]
    
    
    ###########
    # MODEL 2 TRAINING
    ###########
    in_model2.fit(x_all, y_all)
    # again, extract probabilistic predictions for the target data
    predictions2 = in_model2.predict_proba(x_target)
    prob_idx2 = np.where(in_model2.classes_ == 1)[0][0]
    # apply the weighting for model 2 to each prediction
    # note that our weighting for number 2 is (1 - weight[1])
    weight_2 = 1 - weight_1
    class_probs2 = [pred[prob_idx2]*weight_2 for pred in predictions2]
    
    # combine probabilities
    class_probs = np.add(class_probs1, class_probs2)
    
    # and return a matrix of the id's and the corresponding probabilities
    return_mat = [[id_target[idx], class_probs[idx]] \
                    for idx in xrange(len(class_probs))]
                        
    # report our progress before returning
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
    return singleDriverTrainer(target_file, train_file_names, model1, model2,
                               weight_1 = 0.7875,
                               file_subset_size=num_samples_per_training_file)

    

if __name__ == '__main__':
    # initialize the pool for parallelization
    p = Pool()
    # and map our prediction function to the list of data files
    pred_arrays = p.map(parallelWrapper, random.sample(all_files,4))
    
    # now reduce these into a single result
    predictions_combined = reduce(lambda a,b: np.vstack((a,b)), pred_arrays)
    
    # and write to a csv
    df = pd.DataFrame(predictions_combined, columns = ['driver_trip', 'prob'])
    df.to_csv(output_filename, index = False)
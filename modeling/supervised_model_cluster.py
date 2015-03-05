
# standard imports
import copy
import time
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist, squareform
# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
# utility imports
from utils import genListOfCSVs
from utils import bootstrap
# import for parallelization
from multiprocessing import Pool




# let's define our global constants here
# for ease of modification
# path of the summary files
path = '../extracted'
# path for output file name
output_filename = 'solutions_5_clustered.csv'
all_files = genListOfCSVs(path)
# number of training/noise files to use
train_file_count = 4
# specify the number of features, for simplicity
#num_features = min(32, (train_file_count+1)*4)
num_features = 10
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
    data.drop(labels=['avg_velocity_no_0', 'avg_deceleration', 'right_turns_taken', 'avg_velocity',
                  'distance_traveled', 'max_acceleration', 'max_deceleration', 'med_velocity_no_0',
                  'time_spent_cruising', 'time_spent_braking', 'time_spent_accelerating',
                  'avg_right_turn_angle', 'avg_left_turn_angle', 'med_acceleration',
                  'left_turn_fraction', 'sd_left_turn_angle', 'med_right_turn_angle'], axis=1, inplace=True)

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


def singleDriverTrainer2(file_to_classify, training_files, threshold = 0.2,
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

    #upsample target to balance classes
    if len(training_files) > 1:
        num_samples = len(x_target_upsampled) * len(training_files)
        x_target_upsampled, y_target_upsampled = bootstrap(x_target,
                                                           y_target,
                                                           num_samples)

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


def sampling(X_train, y_train):
    """
    This function takes the training data and produces balanced training data with the target
    containing equal numbers of positive and negative classes.
    """

    positive_indexes = [i for i, item in enumerate(y_train) if item == 1]
    negative_indexes = [i for i, item in enumerate(y_train) if item == 0]
    length_train = int(X_train.shape[0]/2.0)

    positive_sample = [random.choice(positive_indexes) for i in xrange(length_train)]
    negative_sample = [random.choice(negative_indexes) for i in xrange(length_train)]
    Xnew = np.vstack((X_train[positive_sample], X_train[negative_sample]))

    ynew_pos = []
    ynew_neg = []
    for i in positive_sample:
        ynew_pos.append(y_train[i])
    for j in negative_sample:
        ynew_neg.append(y_train[j])

    y_train = np.hstack((ynew_pos, ynew_neg))
    return Xnew, y_train



def singleDriverTrainer(file_to_classify, training_files,
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
    x_target, y_, id_target = extractCSV(file_to_classify, target = 1)
    # remove na's


    x_target = np.nan_to_num(x_target)
    y_target = db_scan_clustering(x_target, 3, 101)
    # print len(set(y_target))
    if len(set(y_target)) == 1:
        y_target = np.nan_to_num(y_)

    # now grab the training/noise data
    x_all = copy.copy(x_target)
    y_all = copy.copy(y_target)

    # up-sample target to balance classes, if necessary
    # if len(training_files) > 1:
    #     x_all, y_all = bootstrap(x_all,y_all,len(training_files)*len(x_target))

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
    x_train, y_train = sampling(x_all, y_all)

    # with all of our data, now we can train our model
    in_model.fit(x_train, y_train)

    # now we are ready to provide class probabilities for our predictions
    predictions = in_model.predict_proba(x_target)
    # note that we must extract the index of the class 1 probability
    prob_idx = np.where(in_model.classes_ == 1)[0][0]
    class_probs = [pred[prob_idx] for pred in predictions]
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
    return singleDriverTrainer(target_file, train_file_names, in_model = model)




def db_scan_clustering(x, max_dist, samples):
    D = squareform(pdist(x))
    S = 1 - (D / np.max(D))
    db = DBSCAN(eps=max_dist, min_samples=samples).fit(S)
    if len(set(db.labels_)) == 1:
        l2 = [1*len(db.labels_)]
    else:
        l2 = [0 if i == -1 else 1 for i in db.labels_]




    # [ unicode(x.strip()) if x is not None else '' for x in row ]
    # Number of clusters in labels, ignoring noise if present (noise is labeled as -1).
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return np.asarray(l2)



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

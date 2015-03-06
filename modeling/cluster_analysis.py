__author__ = 'monicameyer'


import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from utils import genListOfCSVs
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import distance
from scipy.spatial.distance import pdist


def extractCSV(file_path, id_column='id_list'):
    """
    Takes in a file path to a given CSV,
    and the name of the id column

    Returns:
        1) x = numpy array, with id's removed
        2) ids = list of id's corresponding to the observations in the x-matrix
    """
    # read in the data
    data = pd.read_csv(file_path, header=0)
    data.drop(labels=['avg_velocity_no_0', 'avg_deceleration', 'right_turns_taken', 'avg_velocity',
              'distance_traveled', 'max_acceleration', 'max_deceleration', 'med_velocity_no_0',
              'time_spent_cruising', 'time_spent_braking', 'time_spent_accelerating',
              'avg_right_turn_angle', 'avg_left_turn_angle', 'med_acceleration',
              'left_turn_fraction', 'sd_left_turn_angle', 'med_right_turn_angle'], axis=1, inplace=True)
    # print data.columns
    # remove the id column
    ids = data.pop(id_column).tolist()

    # create the x-matrix
    x = data.as_matrix()

    return x, ids


def mean_shift_clustering(x):
    D = distance.squareform(distance.pdist(x))
    bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    colors = ([ ([0.4, 1, 0.4],
                 [1, 0.4, 0.4],
                 [0.4, 0.4, 1],
                 [1, 1, 0.4],
                 [1, 0.4, 1],
                 [0.4, 1, 1],
                 [1, 1, 1])[i] for i in labels])

    plt.scatter(x[:, 0], x[:, 1], c=colors)
    for i in range(n_clusters_):
        lines = plt.plot(cluster_centers[i, 0], cluster_centers[i, 1], 'kx')
        plt.setp(lines, ms=15.)
        plt.setp(lines, mew=2.)
    plt.title('Mean Shift, Estimated number of clusters: %d\nSilhouette Coefficient: %0.3f' %
             (n_clusters_, metrics.silhouette_score(D, labels, metric='precomputed')))
    plt.show()
    return


def db_scan_clustering2(x):
    D = distance.squareform(distance.pdist(x))
    S = 1 - (D / np.max(D))
    db = DBSCAN().fit(S)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    colors = ([ ([0.4, 1, 0.4], [1, 0.4, 0.4], [0.4, 0.4, 1],
                 [1, 1, 0.4], [1, 0.4, 1], [0.4, 1, 1],
                 [1, 1, 1])[i] for i in labels])

    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 2
        class_members = [index[0] for index in np.argwhere(labels == k)]
        for index in class_members:
            x_new = x[index]
            markersize = 8
            plt.plot(x_new[0], x_new[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=markersize)

    plt.title('DBSCAN, Estimated number of clusters: %d\nSilhouette Coefficient: %0.3f' %
             (n_clusters_, metrics.silhouette_score(D, labels, metric='precomputed')))
    plt.show()
    return


def db_scan_clustering(x, max_dist, samples):
    D = distance.squareform(pdist(x))
    S = 1 - (D / np.max(D))
    db = DBSCAN(eps=max_dist, min_samples=samples).fit(S)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present (noise is labeled as -1).
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Plot result
    colors = ([ ([0.4, 1, 0.4], [1, 0.4, 0.4], [0.4, 0.4, 1],
                 [1, 1, 0.4], [1, 0.4, 1], [0.4, 1, 1],
                 [1, 1, 1])[i] for i in labels])

    plt.scatter(x[:, 28], x[:, 30], c=colors)

    try:
        sil_score = metrics.silhouette_score(D, labels, metric='precomputed')
        plt.title('DBSCAN, Estimated number of clusters: %d\nNoise points: %d, Silhouette Coefficient: %0.3f' %
                 (n_clusters_, Counter(labels)[-1], sil_score))
        plt.show()
    except:
        print '---------------------error---------------------'

    # plt.show()
    return


def hier_clustering(x):
    knn_graph = kneighbors_graph(x, 30)
    for connectivity in (None, knn_graph):
        for n_clusters in (4, 3, 2):
            plt.figure(figsize=(10, 4))
            for index, linkage in enumerate(('average', 'complete', 'ward')):
                plt.subplot(1, 3, index + 1)
                model = AgglomerativeClustering(linkage=linkage,
                                                connectivity=connectivity,
                                                n_clusters=n_clusters)
                t0 = time.time()
                model.fit(x)
                elapsed_time = time.time() - t0
                plt.scatter(x[:, 0], x[:, 1], c=model.labels_,
                            cmap=plt.cm.spectral)
                plt.title('linkage = %s (time %.2fs)' % (linkage, elapsed_time),
                          fontdict=dict(verticalalignment='top'))
                plt.axis('equal')
                plt.axis('off')

                plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                    left=0, right=1)
                plt.suptitle('n_cluster = %i, connectivity = %r' %
                             (n_clusters, connectivity is not None), size=17)


    plt.show()
    return


if __name__ == '__main__':

    all_files = genListOfCSVs('../extracted')
    corrrr = Counter()
    for file in all_files:
        print file
        x_target, id_target = extractCSV(file)
        x_target = np.nan_to_num(x_target)
        # now grab the training/noise data
        x_all = copy.copy(x_target)

        # mean_shift_clustering(x_all)
        # hier_clustering(x_all)
        # db_scan_clustering2(x_all)
        db_scan_clustering(x_all, 3, 101)



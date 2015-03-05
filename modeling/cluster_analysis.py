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

# from scipy.cluster.vq import vq
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist


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

    # print data.columns
    # remove the id column
    ids = data.pop(id_column).tolist()
    # print data.columns.values
    # create the x-matrix
    x = data.as_matrix()
    # print x.columns
    corr = []
    for i in range(len(x[0])):
        for j in range(len(x[0])):
            if i < j:
                p = pearsonr(x[:, i], x[:, j])[0]
                if p > .7 and p < 1:
                    corr.append([i, j])
    # print np.corrcoef(x[:, 0],x[:, 1])

    return x, ids, corr, data.columns.values


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


def db_scan_clustering(x):
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


def db_scan_clustering_2(x, max_dist, samples):
    D = distance.squareform(pdist(x))
    S = 1 - (D / np.max(D))
    print
    # print pdist(S)
    dist = distance.squareform(pdist(S))
    # print dist.shape
    db = DBSCAN(eps=max_dist, min_samples=samples).fit(S)
    labels = db.labels_
    print Counter(labels)

    # Number of clusters in labels, ignoring noise if present (noise is labeled as -1).
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Plot result
    colors = ([ ([0.4, 1, 0.4], [1, 0.4, 0.4], [0.4, 0.4, 1],
                 [1, 1, 0.4], [1, 0.4, 1], [0.4, 1, 1],
                 [1, 1, 1])[i] for i in labels])

    plt.scatter(x[:, 28], x[:, 30], c=colors)
    try:
        plt.title('DBSCAN, Estimated number of clusters: %d\nSilhouette Coefficient: %0.3f' %
                 (n_clusters_, metrics.silhouette_score(D, labels, metric='precomputed')))
    except:
        print '------------------error------------------'
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

    all_files = genListOfCSVs('extracted')
    corrrr = Counter()
    for file in all_files:
        print file
        x_target, id_target, correlations, names = extractCSV(file)
        for i in correlations:
            corrrr[names[i[0]] + ', ' + names[i[1]]] += 1

        x_target = np.nan_to_num(x_target)
        # now grab the training/noise data
        x_all = copy.copy(x_target)

        # mean_shift_clustering(x_all)
        # hier_clustering(x_all)
        # db_scan_clustering(x_all)
        # db_scan_clustering_2(x_all, 3, 101)
    print corrrr
    for cor in corrrr:
        print cor, corrrr[cor]



__author__ = 'monicameyer'


import copy
import time
import pylab as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import genListOfCSVs
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics


from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift, estimate_bandwidth


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
    data = pd.read_csv(file_path, header=0)
    # remove the id column
    ids = data.pop(id_column).tolist()
    # create the x-matrix
    x = data.as_matrix()
    # and create the corresponding y target values
    y = np.asarray([target] * len(x))
    # now we can return
    return x, y, ids


def compare_kmeans(k):

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x_all)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.title('k-Means for k = ' + str(k))

    for i in range(k):
        subset = x_all[np.where(labels == i)]
        plt.plot(subset[:, 0], subset[:, 1], 'o')
        # plot the centroids
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
        # make the centroid x's bigger
        plt.setp(lines, ms=15.0)
        plt.setp(lines, mew=2.0)
    plt.show()


def kmeans_silhouette(k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x_all)
    labels = kmeans.labels_
    score = silhouette_score(x_all, labels)
    return score

all_files = genListOfCSVs('extracted')

for file in all_files:
    print file
    x_target, y_target, id_target = extractCSV(file, target=1)

    # now grab the training/noise data
    x_all = copy.copy(x_target)
    y_all = copy.copy(y_target)

    try:
        D = distance.squareform(distance.pdist(x_all))
        bandwidth = estimate_bandwidth(x_all, quantile=0.2, n_samples=500)

        ms = MeanShift(bin_seeding=True)
        ms.fit(x_all)
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

        plt.scatter(x_all[:, 0], x_all[:, 1], c=colors)
        for i in range(n_clusters_):
            lines = plt.plot(cluster_centers[i, 0], cluster_centers[i, 1], 'kx')
            plt.setp(lines, ms=15.)
            plt.setp(lines, mew=2.)
        plt.title('Estimated number of clusters: %d\nSilhouette Coefficient: %0.3f' %
                 (n_clusters_, metrics.silhouette_score(D, labels, metric='precomputed')))
        plt.show()
    except:pass


    try:
        knn_graph = kneighbors_graph(x_all, 30)

        for connectivity in (None, knn_graph):
            for n_clusters in (4, 3, 2):
                plt.figure(figsize=(10, 4))
                for index, linkage in enumerate(('average', 'complete', 'ward')):
                    plt.subplot(1, 3, index + 1)
                    model = AgglomerativeClustering(linkage=linkage,
                                                    connectivity=connectivity,
                                                    n_clusters=n_clusters)
                    t0 = time.time()
                    model.fit(x_all)
                    elapsed_time = time.time() - t0
                    plt.scatter(x_all[:, 0], x_all[:, 1], c=model.labels_,
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
    except:pass

    try:
        D = distance.squareform(distance.pdist(x_all))
        S = 1 - (D / np.max(D))

        db = DBSCAN().fit(S)
        core_samples = db.core_sample_indices_
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


        pl.close('all')
        pl.figure(1)
        pl.clf()

        # Black removed and is used for noise instead.
        colors = ([ ([0.4, 1, 0.4],
                     [1, 0.4, 0.4],
                     [0.4, 0.4, 1],
                     [1, 1, 0.4],
                     [1, 0.4, 1],
                     [0.4, 1, 1],
                     [1, 1, 1])[i] for i in labels])

        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
                markersize = 2
            class_members = [index[0] for index in np.argwhere(labels == k)]
            cluster_core_samples = [index for index in core_samples
                                    if labels[index] == k]
            for index in class_members:
                x = x_all[index]
                markersize = 8
                pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=markersize)

        pl.title('Estimated number of clusters: %d\nSilhouette Coefficient: %0.3f' %
                 (n_clusters_, metrics.silhouette_score(D, labels, metric='precomputed')))
        pl.show()
    except:pass



    # try:
    #     # compare_kmeans(2)
    #     # compare_kmeans(3)
    #     # compare_kmeans(4)
    #     # compare_kmeans(5)
    #     silhouettes = []
    #     for i in range(2, 10):
    #         silhouettes.append(kmeans_silhouette(i))
    #     print silhouettes.index(max(silhouettes))+2
    #
    #     plt.plot(range(2,10), silhouettes, 'ro-', lw=2)
    #     plt.title('Silhouette coefficient plot')
    #     plt.xlabel('Number of clusters')
    #     plt.ylabel('Silhouette coefficient')
    #     plt.ylim(0, 1)
    #     plt.xlim(1, 10)
    #     plt.show()
    # except:pass









# from scipy.cluster.vq import vq
# Z = [vq(X,cent) for cent in centroids]
# avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]
# K = range(1, 10)
# KM = [kmeans(x_all, k) for k in K]
# centroids = [cent for (cent, var) in KM]

# clu = KMeans(n_clusters=2)
# labels = clu.fit_predict(x_all)

# alternative: scipy.spatial.distance.cdist

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import BaggingClassifier
# from sklearn.linear_model import LogisticRegression
#
# x_target, y_target, id_target = extractCSV('extracted/1_summary.csv', target=0)
# x_other, y_other, id_other = extractCSV('extracted/2_summary.csv', target=1)
#
# # now grab the training/noise data
# x_all = copy.copy(x_target)
# y_all = copy.copy(y_target)
# x_all = np.concatenate((x_all, x_other))
# y_all = np.concatenate((y_all, y_other))
# x_all = np.nan_to_num(x_all)
# y_all = np.nan_to_num(y_all)
#
#
# for j in range(2,30):
#     anova_filter = SelectKBest(f_classif, k=j)
#     clf = BaggingClassifier(LogisticRegression())
#     anova_log = Pipeline([('anova', anova_filter), ('log', clf)])
#     anova_log.set_params(anova__k=10).fit(x_all, y_all)
#
#     prediction = anova_log.predict(x_all)
#     print j, anova_log.score(x_all, y_all)
#
# print x_all.shape
#
# clu = AgglomerativeClustering(n_clusters=3)
# labels = clu.fit_predict(x_all)
# colors = ([([0.4, 1, 0.4], [1, 0.4, 0.4], [0.1, 0.8, 1])[i] for i in labels])
# plt.scatter(x_all[:, 0], x_all[:, 1], c=colors)
# plt.show()
#
#
# from scipy.spatial.distance import pdist, squareform
# import scipy.cluster.hierarchy as sch
# import scipy
# # euclidean default for pdist:
# dists = sch.distance.pdist(x_all)
# print dists
# clus = sch.linkage(dists)
#
# print sch.dendrogram(clus);
#
# # get the flat cluster "labels"
# flats = sch.fcluster(clus, 0.5, 'distance')
# print flats
#
# compl = sch.linkage(dists, method='complete')
# print sch.dendrogram(compl);
#
# centr = sch.linkage(x_all, method='centroid')
# print sch.dendrogram(centr)

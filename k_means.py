from sklearn.cluster import KMeans
import numpy as np
from object import *
from treelib import *


def hi_kmeans(_first_node, _des_database_list, _b, _depth):  # need to add depth

    # print('tot features', len(_des_database_list))
    X = []
    for i in range(len(_des_database_list)):
        X.append(_des_database_list[i].vector)

    X_new = np.array(X)

    kmeans = KMeans(n_clusters=_b, random_state=0).fit(X_new)
    kmeans_labels = kmeans.labels_
    #print(len(kmeans_labels))

    clusters = [[] for i in range(_b)]
    kmeans_centroids = kmeans.cluster_centers_
    centroids = kmeans_centroids.tolist()

    '''
    for m in range(_b):
        for x, y in zip(X_new, kmeans_labels):
            if y == m:
                clusters[m].append(keypoint_with_id(x, _des_database_list[X.index(x)].id))
        # print ("m",len(clusters[m]))

    centroids  = kmeans.cluster_centers_
    '''

    for a in range(len(kmeans_labels)):  # total numbers of descriptors
        tmp_list = []
        if len(clusters[kmeans_labels[a]]) > 0:
            tmp_list.extend(clusters[kmeans_labels[a]])
        tmp_list.append(keypoint_with_id(_des_database_list[a].vector, _des_database_list[a].id))
        clusters[kmeans_labels[a]] = tmp_list

    # build tree with recursive method
    if _depth > 0:
        _depth -= 1
        for m in range(_b):
            _child = Tree(clusters[m], centroids[m])
            _first_node.addChild(_child)
            hi_kmeans(_child, clusters[m], _b, _depth)
        # _first_node.nestedTree()

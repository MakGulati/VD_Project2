from sklearn.cluster import KMeans
import numpy as np
from object import *
from treelib import *
from math import *


def hi_kmeans(_first_node, _des_database_list, _b, _depth, _n_documents):  # need to add depth

    descriptors = []
    for i in range(len(_des_database_list)):
        descriptors.append(_des_database_list[i].vector)

    descriptors_new = np.array(descriptors) # transform list to np array for kmeans
    
    if (len(descriptors) > _b):
        kmeans = KMeans(n_clusters=_b, random_state=0).fit(descriptors_new) # kmeans clustering algorithm
        kmeans_labels = kmeans.labels_ # obtaining the labels of the clusters
    
        clusters = [[] for i in range(_b)]
        centroids = kmeans.cluster_centers_ # computing centroid for each cluster
    
        '''
        for m in range(_b):
            for x, y in zip(X_new, kmeans_labels):
                if y == m:
                    clusters[m].append(keypoint_with_id(x, _des_database_list[X.index(x)].id))
            # print ("m",len(clusters[m]))
    
        centroids  = kmeans.cluster_centers_
        '''
    
        # populating the clusters labeled with the descritpors and corresponding id
        for a in range(len(kmeans_labels)):  # total numbers of descriptors
            tmp_list = []
            if len(clusters[kmeans_labels[a]]) > 0: #if you have already elements in that cluster
                tmp_list.extend(clusters[kmeans_labels[a]]) # extend because want to append each elements instead of appending the list 
            tmp_list.append(keypoint_with_id(_des_database_list[a].vector, _des_database_list[a].id))
            clusters[kmeans_labels[a]] = tmp_list
    
        # compute td-idf weights table for each node
        tfidf_scores = [[] for i in range(_b)]
        tmp_list_id = [[] for i in range(_b)]
        tf = [[] for i in range(_b)]
    
        for i in range(_b):
            for j in range(len(clusters[i])):
                tmp_list_id[i].append(clusters[i][j].id)
            print("hi")
            print(set(tmp_list_id[0]))
            for k in range(_n_documents):
                tf[i].append(tmp_list_id[i].count(k) / (len(clusters[i])))
            idf = log2(_n_documents / np.count_nonzero(tf[i])) # compute id
            tfidf_scores[i] = (np.array(tf[i]) * idf).tolist()
    
    
        # build tree with recursive method
        if _depth > 0: # only if there is still depth
            _depth -= 1
            for m in range(_b):
                _child = Tree(clusters[m], centroids[m], tfidf_scores[m]) # child
                _first_node.addChild(_child) # adding the child to the parent
                hi_kmeans(_child, clusters[m], _b, _depth,_n_documents) # kmeans clustering on each child

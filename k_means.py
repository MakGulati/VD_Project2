from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from object import *

def hi_kmeans(_first_node, _des_database_list, _b, _depth): #need to add depth

    # print('tot features', len(_des_database_list))
    X = []
    for i in range(len(_des_database_list)):
        X.append(_des_database_list[i].vector)

    X_new = np.array(X)

        
    kmeans = KMeans(n_clusters=_b, random_state=0).fit(X_new)
    #y_kmeans = clusters.predict(X_new)
    #print(y_kmeans)
    kmeans_labels = kmeans.labels_
    #print(kmeans_labels)
    #print(X.shape[0])

    clusters = [[] for i in range (_b)]

    for m in range(_b):
        for x, y in zip(X_new, kmeans_labels):
            if y==m:
                clusters[m].append(keypoint_with_id(x, _des_database_list[X_new.index(x)].id))
        # print ("m",len(clusters[m]))

    # for i in range(len(_des_database_list)):
    #     X.append(_des_database_list[i].vector)

    #build tree
    while(_depth > 0):
        _depth -= 1
        for m in range (_b):
            _child = Tree(clusters[m])
            _first_node.addChild(_child)
            hi_kmeans(_child, clusters[m], _b, _depth)
        _first_node.nestedTree()







    #plt.scatter(X_new[:, 0], X_new[:, 127], c=kmeans_labels, s=50, cmap='viridis')

    #centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 127],c='red',alpha=0.9)
    # plt.show()

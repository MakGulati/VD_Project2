from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def hi_kmeans(_object, _b):
    X=_object[0].des_mat
    for i in range(1,50):
        X = np.vstack((X,_object[i].des_mat))

    clusters = KMeans(n_clusters=_b, random_state=0).fit(X)
    y_kmeans=clusters.predict(X)
    print(X.shape[0])
    plt.scatter(X[:, 0], X[:, 127], c=y_kmeans, s=50, cmap='viridis')

    centers = clusters.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 127],c='red',alpha=0.9)
    plt.show()
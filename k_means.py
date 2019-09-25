from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def hi_kmeans(_object, _b):
    X = _object.des_mat
    clusters = KMeans(n_clusters=_b, random_state=0).fit(X)

    k_means=clusters.predict(X)

    # plt.scatter(X[:, 0], X[:, 127], c=k_means, s=50, cmap='viridis')

    centers = clusters.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 127],c='red',alpha=0.9)
    # plt.show()
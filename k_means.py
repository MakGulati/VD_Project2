from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[6, 0], [1, 4], [1, 0],
              [10, 1], [0, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)

y_kmeans=kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='green',alpha=0.9)
plt.show()
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

descriptors = [[1,2,3],[3,4,5], [0,2,3],[1,2,3],[4,5,6],[0,1,2],[4,5,6]]

descriptors = np.float32(descriptors)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans ---------- data, num of clusters, criteria, attempts, flags
compactness, labels, centers = cv2.kmeans(descriptors, 3, None, criteria, 10, flags)

print(labels)

for a in range(len(labels)):
    print(labels[a][0])
'''

from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1,2,3],[3,4,5], [0,2,3],[1,2,3],[4,5,6],[0,1,2],[4,5,6]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
print(labels)

for a in range(len(labels)):
    print(labels[a])

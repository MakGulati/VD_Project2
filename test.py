## ASVD - PROJECT 2 - September 2019
## Mayank Gulati & Federico Favia

import numpy as np
import cv2
from object import *
from k_means import *

## 2 IMAGE FEATURE EXTRACTION
# (a) Extract few hundreds features from each database image and combine the ones for same object, avg n°features per database object

n_keypoints = 250  # strongest keypoints to keep

# directory path with database images
dir_path_database = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"
#dir_path_database = 'Data2/server/obj'

tot_features_database = 0  # counting total features of database for retrieving average
database_des = {}  # dictionary of database objects containing descriptors

for i in range(50):  # 250 images with 50 objects, 3 images per object read at time
    img1 = cv2.imread(dir_path_database + str(i + 1) + "_1.jpg", cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create(n_keypoints)
    kp1, des1 = sift.detectAndCompute(img1, None)
    des = des1
    img2 = cv2.imread(dir_path_database + str(i + 1) + "_2.jpg", cv2.IMREAD_GRAYSCALE)
    kp2, des2 = sift.detectAndCompute(img2, None)
    img3 = cv2.imread(dir_path_database + str(i + 1) + "_3.jpg", cv2.IMREAD_GRAYSCALE)
    kp3, des3 = sift.detectAndCompute(img3, None)


    for j in [des2, des3]: #comparing same building images for removing redundant SIFT descritpors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, j, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # if it is a match
                good.append(m.queryIdx)  # saving index for removing it
        m2 = np.delete(j, good, 0)  # removing matching features in same object
        des = np.vstack((des, m2))

    database_des[i] = object(i + 1, des)
    tot_features_database += database_des[i].__len__()

# avg n°feature extracted per database object
avg_feature_database_object = tot_features_database / len(database_des)
print('Avg # feature per database object = ', avg_feature_database_object)

'''
# (b) Extract few hundreds features from each query image and save them separately, avg n°features per query object

# directory path with query images
# dir_path_query = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/client/obj"
dir_path_query ='Data2/client/obj'

tot_features_query = 0 # counting total features of database for retrieving average
query_des = {} # dictionary of query objects containing descriptors

for i in range(50): # 50 query images
    img_i = cv2.imread(dir_path_query+str(i+1)+"_t1.jpg", cv2.IMREAD_GRAYSCALE)
#   sift_i = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.19, edgeThreshold = 9.3)
    sift_i = cv2.xfeatures2d.SIFT_create(n_keypoints)
    kp_i, des_i = sift_i.detectAndCompute(img_i, None)

    query_des[i] = object(i+1, des_i)
    tot_features_query += query_des[i].__len__()

# avg n°feature extracted per query object
avg_feature_query_object = tot_features_query / len(query_des)
print("Avg # feature per query object = ", avg_feature_qchild02.data(str(202))uery_object)
'''

# 3 VOCABULARY TREE CONSTRUCTION

hi_kmeans(database_des,3)

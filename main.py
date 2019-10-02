## ASVD - PROJECT 2 - September 2019
## Mayank Gulati & Federico Favia

import cv2
from hi_k_means import *
from treelib import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #remove DeprecationWarning

## 2 IMAGE FEATURE EXTRACTION
# (a) Extract few hundreds features from each database image and combine the ones for same object, avg n°features per database object

n_documents = 50  # n of documents (buildings) presents in database
n_queries = 50  # n of query images
n_keypoints = 350  # strongest keypoints to keep 350
nndr_thresh = 0.80  # thresh for nndr SIFT match

# directory path with database images
# dir_path_database = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"
dir_path_database = 'Data2/server/obj'

# merging features for database images
tot_features_database = 0  # counting total features of database for retrieving average
des_database = {}  # dictionary of database objects containing descriptors

for i in range(n_documents):  # 250 images with 50 buildings (documents), 3 images per object read at time
    img1 = cv2.imread(dir_path_database + str(i + 1) + "_1.jpg", cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create(n_keypoints)
    kp1, des1 = sift.detectAndCompute(img1, None)
    des = des1
    img2 = cv2.imread(dir_path_database + str(i + 1) + "_2.jpg", cv2.IMREAD_GRAYSCALE)
    kp2, des2 = sift.detectAndCompute(img2, None)
    img3 = cv2.imread(dir_path_database + str(i + 1) + "_3.jpg", cv2.IMREAD_GRAYSCALE)
    kp3, des3 = sift.detectAndCompute(img3, None)

    for j in [des2, des3]:  # comparing same building images for removing redundant SIFT descritpors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, j, k=2)
        good = []
        for m, n in matches:
            if m.distance < nndr_thresh * n.distance:  # if it is a match
                good.append(m.queryIdx)  # saving index for removing it
        m2 = np.delete(j, good, 0)  # removing matching features in same object
        des = np.vstack((des, m2))

    des_database[i] = keypoints_mat_with_id(des, i)
    tot_features_database += des_database[i].__len__()

# Avg n° feature extracted per database object
avg_feature_database_object = tot_features_database / len(des_database)
print('Avg # feature per database object = ', avg_feature_database_object)


# (b) Extract few hundreds features from each query image and save them separately, avg n°features per query object

# directory path with query images
# dir_path_query = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/client/obj"
dir_path_query ='Data2/client/obj'

tot_features_query = 0  # counting total features of database for retrieving average
des_query = {}  # dictionary of query objects containing descriptors

factors = [1, 0.9, 0.7, 0.5]  # to feed the vocabulary with less number of SIFT descriptors (90%, 70% and 50%)

for i in range(n_queries):  # 50 query images
    img_i = cv2.imread(dir_path_query + str(i + 1) + "_t1.jpg", cv2.IMREAD_GRAYSCALE)
    sift_i = cv2.xfeatures2d.SIFT_create(int(round(factors[0] * n_keypoints)))
    kp_i, des_i = sift_i.detectAndCompute(img_i, None)

    des_query[i] = keypoints_mat_with_id(des_i, i)
    tot_features_query += des_query[i].__len__()

# Avg n° feature extracted per query object
avg_feature_query_object = tot_features_query / len(des_query)
print("Avg # feature per query object = ", avg_feature_query_object)

# 3 VOCABULARY TREE CONSTRUCTION and 4 QUERYING

# Assign document id to each descriptor and creating list of descriptors
des_database_list = []

for i in range(n_documents):
    for j in range(des_database[i].__len__()):
        des_database_list.append(keypoint_with_id(des_database[i].get_des(j), des_database[i].doc_id))

# setting root of tree (same for every tree)
parent_node = Tree(des_database_list)

# building 1st tree (b=4, depth=3)
b = 4  # n of branches (clusters) in each level of tree
depth = 5  # n of levels of tree
hi_kmeans(parent_node, des_database_list, b, depth, n_documents)  # b is number of clusters, depth is number of levels

print("Tree has been built! Now querying...")

counter_t1 = 0
counter_t5 = 0

for i in range(n_queries):
    accu_list = [0 for s in range(n_documents)]
    top5=[]
    for j in range(des_query[i].__len__()):
        tmp_parent_node = parent_node

        for d in range(depth):
            first_tree = tmp_parent_node.getChildren()

            if len(first_tree) > 0:
                euclid_dist =[]
                for node in range(b):
                    euclid_dist.append(np.linalg.norm(des_query[i].get_des(j) - np.array(first_tree[node].centroid)))

                closer_child_index = euclid_dist.index(min(euclid_dist))
                tmp_parent_node = first_tree[closer_child_index]

        # summing up tfidf scores of leaf nodes
        # accu_list = list(map(add, accu_list, parent_node.tfidf_score)) # list of 50 elements (doc id)
        best_leaf_node = tmp_parent_node
        accu_list = (np.add(accu_list, best_leaf_node.tfidf_score)) # list of 50 elements (doc id)
        accu_list = accu_list.tolist()

    top1 = accu_list.index(max(accu_list))
    top5_items = sorted(accu_list, reverse=True)[:5]
    for p in range(5):
        top5.append(accu_list.index(top5_items[p]))


    if top1 == i: # image correctly classified?
        counter_t1 += 1

    if i in top5:
        counter_t5 += 1

# Avg top-1 recall rate
avg_recall_rate_t1 = counter_t1 / n_queries
print("Avg recall rate top-1 = ", avg_recall_rate_t1)

# Avg top-5 recall rate
avg_recall_rate_t5 = counter_t5 / n_queries
print("Avg recall rate top-5 = ", avg_recall_rate_t5)








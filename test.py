## ASVD - PROJECT 2
## Mayank Gulati and Federico Favia

import numpy as np
import cv2
import csv


import matplotlib.pyplot as plt



## IMAGE FEATURE EXTRACTION

#img_dir = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/"

#des = []

#data_path = os.path.join(img_dir,'*g')
#files = glob.glob(data_path)
#data = []
#m = np.empty()

#for f1 in files:
'''
dir_path_database = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"

for i in range(2):
    final = np.empty((2*750, 128))

    for j in range(3):
        img = cv2.imread(dir_path_database+str(i+1)+"_"+str(j+1)+".jpg", cv2.IMREAD_GRAYSCALE)
        #data.append(img)
        #cv2.imshow('im', img)
        #cv2.waitKey(0)
        #sift = cv2.xfeatures2d.SIFT_create(249) #strongest 150
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08, edgeThreshold=8)
        kp1, des1 = sift.detectAndCompute(img, None)
        np.concatenate((final, des1), axis = 0)

new = final.reshape(-1,750,128)
'''

#print("printing lists in new line")
#print(final.shape)
#print(new.shape)

# (b) Extract few hundreds features from each query image and save them separately, avg n°features per query object
dir_path_query = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/client/obj"

query_des = []
tot_features = 0

for i in range(50):
    query_des.append("n" + str(i+1))
    img_i = cv2.imread(dir_path_query+str(i+1)+"_t1.jpg", cv2.IMREAD_GRAYSCALE)
    sift_i = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.19, edgeThreshold = 9.3)
    kp_i, des_i = sift_i.detectAndCompute(img_i, None)
    query_des[i] = des_i
    #np.savetxt('data' '.csv', (col1_array, col2_array, col3_array), delimiter=',')

    tot_features = tot_features + query_des[i].shape[0]

#avg n°feature extracted per query object
avg_feature_query_object = tot_features / len(query_des)
print("Avg # feature per object = ", avg_feature_query_object)


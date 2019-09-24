## ASVD - PROJECT 2
## Mayank Gulati & Federico Favia

import numpy as np
import cv2
import matplotlib.pyplot as plt

# (a)

dir_path_database = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"
#dir_path_database = 'Data2/server/obj'
des_comparison = []

for i in range(1): #50 objects, 3 images per time

    img1 = cv2.imread(dir_path_database + str(i+1) + "_1.jpg", cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.2, edgeThreshold = 9.3)
    kp1, des1 = sift.detectAndCompute(img1, None)
    des = des1
    img2 = cv2.imread(dir_path_database + str(i+1) + "_2.jpg", cv2.IMREAD_GRAYSCALE)
    #sift2 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.2, edgeThreshold = 9.3)
    kp2, des2 = sift.detectAndCompute(img2, None)
    des_comparison[i] = des2
    img3 = cv2.imread(dir_path_database + str(i+1) + "_3.jpg", cv2.IMREAD_GRAYSCALE)
    #sift3 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.2, edgeThreshold=9.3)
    kp3, des3 = sift.detectAndCompute(img3, None)
    des_comparison[1] = des3

    for j in range(2): #comparison between first images and two corresponding to delete redundant SIFT features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des_comparison[j], k=2)
        good = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m.queryIdx) #indexes of good matches
        m2 = np.delete(des_comparison[j], good, axis=0) #delete from descriptors the one that are good matches
        des = np.vstack((des, m2))

# bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1, des3, k=2)
    #good = []
    #for m, n in matches:
        #    if m.distance < 0.75 * n.distance:
    #       good.append(m.queryIdx)
    #m2 = np.delete(des3, good, axis=0)
    #des = np.vstack((des, m2))
    

# new = final.reshape(-1,750,128)

#print("printing lists in new line")
print(des.shape)




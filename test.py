import numpy as np

import cv2

import matplotlib.pyplot as plt





#for f1 in files:
# dir = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"
dir='Data2/server/obj'
for i in range(1):


    img1 = cv2.imread(dir+str(i+1)+"_1.jpg", cv2.IMREAD_GRAYSCALE)
    sift1 = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2, edgeThreshold=9.3)
    kp1, des1 = sift1.detectAndCompute(img1, None)
    des=des1
    img2 = cv2.imread(dir + str(i+1) + "_2.jpg", cv2.IMREAD_GRAYSCALE)
    sift2 = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2, edgeThreshold=9.3)
    kp2, des2 = sift2.detectAndCompute(img2, None)
    img3 = cv2.imread(dir + str(i + 1) + "_3.jpg", cv2.IMREAD_GRAYSCALE)
    sift3 = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2, edgeThreshold=9.3)
    kp3, des3 = sift3.detectAndCompute(img3, None)

    for j in [des2,des3]:

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,j, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m.queryIdx)
        m2 = np.delete(j, good,0)
        des=np.vstack((des,m2))

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des3, k=2)
    #
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append(m.queryIdx)
    # m2 = np.delete(des3, good, axis=0)
    # des = np.vstack((des, m2))
    


# new = final.reshape(-1,750,128)

#print("printing lists in new line")
print(des.shape)




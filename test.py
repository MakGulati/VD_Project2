'''import cv2 as cv
import matplotlib.pyplot as plt
i = cv.imread('D:/Federico/Desktop/test/obj1_5.jpg')
cv.imshow('img',i)
cv.waitKey(0)
'''

import numpy as np
import os
import glob
import imageio
import cv2 as cv

import matplotlib.pyplot as plt

img_dir = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/"


des = []

data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#data = []
#m = np.empty()

#for f1 in files:
dir = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"

for i in range(2):
    final = np.empty((2*750, 128))

    for j in range(3):
        img = cv.imread(dir+str(i+1)+"_"+str(j+1)+".jpg", cv.IMREAD_GRAYSCALE)
        #data.append(img)
        #cv.imshow('im', img)
        #cv.waitKey(0)
        sift = cv.xfeatures2d.SIFT_create(249) #strongest 150
        sift = cv.xfeatures2d.SIFT_create(contrastThreshold=)
        kp1, des1 = sift.detectAndCompute(img, None)
        np.concatenate((final, des1), axis = 0)

new = final.reshape(-1,750,128)

#print("printing lists in new line")
print(final.shape)
print(new.shape)



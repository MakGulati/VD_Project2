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
m = np.empty([160, 128])

for f1 in files:
    img = cv.imread(f1, cv.IMREAD_GRAYSCALE)
    #data.append(img)
    #cv.imshow('im', img)
    #cv.waitKey(50)
    sift = cv.xfeatures2d.SIFT_create(160) #strongest 150
    kp1, des1 = sift.detectAndCompute(img, None)
    for i in range(160):
        m[i] = des1[i]
        #print(des1[0])

print("printing lists in new line")
print(m.shape)

#print(*des, sep="\n")

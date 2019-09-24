## ASVD - PROJECT 2
## Mayank Gulati & Federico Favia

import numpy as np
import cv2
import matplotlib.pyplot as plt

# (a)
class object():
    def __init__(self,_obj_id,_des_id):
        self.obj_id=_obj_id
        self.des_id=_des_id
    def __len__(self):
        return(self.des_id.shape[0])

    def get_des(self,_pos_id):
        return (self.des_id[_pos_id])

    def __del__(self):
        pass


#dir_path_database = "D:/Federico/Documents/Federico/Uni Trento/03 Magistrale EIT/02 EIT VCC 2019-20/1st period/Analysis and Search of Visual Data EQ2425/Projects/Project 2/Data2/server/obj"
dir_path_database = 'Data2/server/obj'
counter = 0
abc={}
for i in range(50): #50 objects, 3 images per time
    img1 = cv2.imread(dir_path_database + str(i+1) + "_1.jpg", cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create(250) #strongest 250
    kp1, des1 = sift.detectAndCompute(img1, None)
    des = des1
    img2 = cv2.imread(dir_path_database + str(i+1) + "_2.jpg", cv2.IMREAD_GRAYSCALE)
    kp2, des2 = sift.detectAndCompute(img2, None)
    img3 = cv2.imread(dir_path_database + str(i+1) + "_3.jpg", cv2.IMREAD_GRAYSCALE)
    kp3, des3 = sift.detectAndCompute(img3, None)

    for j in [des2, des3]:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, j, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: #if it is a match
                good.append(m.queryIdx) #saving index for removing it
        m2 = np.delete(j, good, 0) #removing matching features in same object
        des = np.vstack((des, m2))

    abc[i]=object(i+1, des)
    counter+=abc[i].__len__()




    
    
print(abc[1].des_id)
#avg n°feature extracted per database object
avg_feature_database_object = counter / 50
print('Avg # feature per object = ', avg_feature_database_object)







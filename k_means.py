from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def hi_kmeans(_des_database_list, _b): #need to add depth
    '''    X=_object[0].des_mat
    for i in range(1,50):
        X = np.vstack((X,_object[i].des_mat))
    '''
    print('tot features', len(_des_database_list))
    
    X=[]
    for i in range(len(_des_database_list)):
        X.append(_des_database_list[i].vector)
    
    X_new=np.array(X)

        
    kmeans = KMeans(n_clusters=_b, random_state=0).fit(X_new)
    #y_kmeans = clusters.predict(X_new)
    #print(y_kmeans)
    kmeans_labels = kmeans.labels_
    #print(kmeans_labels)
    #print(X.shape[0])

    clusters = [[] for i in range (_b)]

    for m in range(_b):
        for x, y in zip(X_new, kmeans_labels):
            if y==m:
                clusters[m].append(x)
        print ("m",len(clusters[m]))







    #plt.scatter(X_new[:, 0], X_new[:, 127], c=kmeans_labels, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 127],c='red',alpha=0.9)
    # plt.show()


# coding: utf-8

# In[20]:

print(__doc__)

# K-means, Sergi B. Exercise of Percepcion // Master on Robotics Eurecat-Uvic

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.cluster import KMeans
from sklearn import datasets
from math import floor
from math import ceil

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[21]:

# Cojemos los maximos de cada columna y los minimos y redondeamos segun
# nos convenga para que esten dentro de los valores del dataset.

x_max=X.max(axis=0)
X_max=max(x_max)

x_min=X.min(axis=0)
X_min=min(x_min)

limit=[X_max,X_min]

min_lim = limit[1]
min_lim = ceil(min_lim)
max_lim = limit[0]
max_lim = floor(max_lim)


# In[22]:

# Inicilizamos k numeros de clusters, cantidad maxima iteraciones.
# Creamos los centros dentro de los valores calculados en el paso anterior

nMax_iter = 100
k = 3

x_coor = np.random.randint(min_lim, max_lim, (1,k))
y_coor = np.random.randint(min_lim, max_lim, (1,k))


# In[23]:

plt.figure(0)
plt.plot(x_coor,y_coor, '.', markersize=10)
plt.xlim(min_lim-1,max_lim+1)
plt.ylim(min_lim-1,max_lim+1)
plt.annotate('Cluster 1', xy=(x_coor[:,0], y_coor[:,0]), xytext=(x_coor[:,0]+1, y_coor[:,0]+1),fontsize=10,
            arrowprops=dict(facecolor='blue', shrink=0.25),
            )
plt.annotate('Cluster 2', xy=(x_coor[:,1], y_coor[:,1]), xytext=(x_coor[:,1]+1, y_coor[:,1]+1),fontsize=10,
            arrowprops=dict(facecolor='green', shrink=0.25),
            )
plt.annotate('Cluster 3', xy=(x_coor[:,2], y_coor[:,2]), xytext=(x_coor[:,2]+1, y_coor[:,2]+1),fontsize=10,
            arrowprops=dict(facecolor='red', shrink=0.25),
            )
plt.legend( ["Clusters[0]"])
plt.grid(True)
plt.show()


# In[24]:

Clusters = [(int(x_coor[:,0]), int(y_coor[:,0])),
            (int(x_coor[:,1]), int(y_coor[:,1])),
            (int(x_coor[:,2]), int(y_coor[:,2]))]
print Clusters


# In[25]:

print "First two dimensions of the iris dataset"
print

dim = 2
X_Q2 = X[:,:dim]
d_min = np.zeros((k,X_Q2.shape[0]))


# In[26]:

plt.figure(1)
plt.plot(x_coor,y_coor, '.', markersize=10)
plt.plot(X_Q2[:,0],X_Q2[:,1], 'c.')
plt.xlim(min_lim-1,max_lim+1)
plt.ylim(min_lim-1,max_lim+1)
plt.annotate('Cluster 1', xy=(x_coor[:,0], y_coor[:,0]), xytext=(x_coor[:,0]-2, y_coor[:,0]+1),fontsize=10,
            arrowprops=dict(facecolor='blue', shrink=0.25),
            )
plt.annotate('Cluster 2', xy=(x_coor[:,1], y_coor[:,1]), xytext=(x_coor[:,1]+1, y_coor[:,1]+1),fontsize=10,
            arrowprops=dict(facecolor='green', shrink=0.25),
            )
plt.annotate('Cluster 3', xy=(x_coor[:,2], y_coor[:,2]), xytext=(x_coor[:,2]+1, y_coor[:,2]+1),fontsize=10,
            arrowprops=dict(facecolor='red', shrink=0.25),
            )
plt.legend(["Clusters[0]" ,"Data"])
plt.grid(True)
plt.show()


# In[27]:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=nMax_iter).fit(X_Q2)
centroids = kmeans.cluster_centers_

centroids


# In[28]:

Cxy = np.hsplit(centroids, 2)


# In[29]:

plt.figure(2)
plt.plot(Cxy[0],Cxy[1], 'kp', label="Centroides" , markersize=10)
plt.plot(x_coor,y_coor, '.', markersize=8)
plt.plot(X_Q2[:,0],X_Q2[:,1], 'y.')
plt.xlim(min_lim-1,max_lim+1)
plt.ylim(min_lim-1,max_lim+1)
plt.annotate('Cluster 1', xy=(x_coor[:,0], y_coor[:,0]), xytext=(x_coor[:,0]-2, y_coor[:,0]+1),fontsize=10,
            arrowprops=dict(facecolor='blue', shrink=0.25),
            )
plt.annotate('Cluster 2', xy=(x_coor[:,1], y_coor[:,1]), xytext=(x_coor[:,1]+1, y_coor[:,1]+1),fontsize=10,
            arrowprops=dict(facecolor='green', shrink=0.25),
            )
plt.annotate('Cluster 3', xy=(x_coor[:,2], y_coor[:,2]), xytext=(x_coor[:,2]+1, y_coor[:,2]+1),fontsize=10,
            arrowprops=dict(facecolor='red', shrink=0.25),
            )
plt.legend(["Centroides","Clusters[0]" ,"Data"])
plt.grid(True)
plt.show()


# In[30]:

clusters = np.zeros((k,X_Q2.shape[0]))
d_min = np.argmin(clusters,0)
y_pred = np.array([centroids[d_min[i],:] for i in range(X_Q2.shape[0])]) 

print "Global MSE is:", mse(X_Q2,y_pred)


# In[31]:

for j in range(k):
    
    if X_Q2[d_min==j,:].any():
        print "MSE Cluster", j, "is:", mse(X_Q2[d_min==j,:],y_pred[d_min == j,:])


# In[32]:

print "Four dimensions of the iris dataset"
print

dim = 4
X_Q4 = X[:,:dim]
d_min = np.zeros((k,X_Q4.shape[0]))


# In[33]:

plt.figure(3)
plt.plot(x_coor,y_coor, 'b.', markersize=10)
plt.plot(X_Q4[:,0],X_Q4[:,1], 'y.')
plt.plot(X_Q4[:,2],X_Q4[:,3], 'y.')
plt.xlim(min_lim-1,max_lim+1)
plt.ylim(min_lim-1,max_lim+1)
plt.annotate('Cluster 1', xy=(x_coor[:,0], y_coor[:,0]), xytext=(x_coor[:,0]-2, y_coor[:,0]+1),fontsize=10,
            arrowprops=dict(facecolor='blue', shrink=0.25),
            )
plt.annotate('Cluster 2', xy=(x_coor[:,1], y_coor[:,1]), xytext=(x_coor[:,1]+1, y_coor[:,1]+1),fontsize=10,
            arrowprops=dict(facecolor='green', shrink=0.25),
            )
plt.annotate('Cluster 3', xy=(x_coor[:,2], y_coor[:,2]), xytext=(x_coor[:,2]+1, y_coor[:,2]+1),fontsize=10,
            arrowprops=dict(facecolor='red', shrink=0.25),
            )
plt.legend( ["Clusters[0]" ,"Data"])
plt.grid(True)
plt.show()


# In[34]:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=nMax_iter).fit(X_Q4)
centroids = kmeans.cluster_centers_

centroids


# In[35]:

Cxy2 = np.hsplit(centroids, 4)
Cxy2


# In[36]:

plt.figure(4)
plt.plot(Cxy2[0],Cxy2[1], 'kp', label="Centroides" , markersize=10)
plt.plot(Cxy2[2],Cxy2[3], 'kp', label="Centroides" , markersize=10)
plt.plot(x_coor,y_coor, '.', markersize=8)
plt.plot(X_Q4[:,0],X_Q4[:,1], 'y.')
plt.plot(X_Q4[:,2],X_Q4[:,3], 'y.')
plt.xlim(min_lim-1,max_lim+1)
plt.ylim(min_lim-1,max_lim+1)
plt.annotate('Cluster 1', xy=(x_coor[:,0], y_coor[:,0]), xytext=(x_coor[:,0]-2, y_coor[:,0]+1),fontsize=10,
            arrowprops=dict(facecolor='blue', shrink=0.25),
            )
plt.annotate('Cluster 2', xy=(x_coor[:,1], y_coor[:,1]), xytext=(x_coor[:,1]+1, y_coor[:,1]+1),fontsize=10,
            arrowprops=dict(facecolor='green', shrink=0.25),
            )
plt.annotate('Cluster 3', xy=(x_coor[:,2], y_coor[:,2]), xytext=(x_coor[:,2]+1, y_coor[:,2]+1),fontsize=10,
            arrowprops=dict(facecolor='red', shrink=0.25),
            )
plt.legend(["Centroides", "Centroides", "Clusters[0]" ,"Data"])
plt.grid(True)
plt.show()


# In[37]:

clusters = np.zeros((k,X_Q4.shape[0]))
d_min = np.argmin(clusters,0)
y_pred = np.array([centroids[d_min[i],:] for i in range(X_Q4.shape[0])]) 

print "Global MSE is:", mse(X_Q4,y_pred)


# In[38]:

for j in range(k):
    
    if X_Q4[d_min==j,:].any():
        print "MSE Cluster", j, "is:", mse(X_Q4[d_min==j,:],y_pred[d_min == j,:])


# In[ ]:




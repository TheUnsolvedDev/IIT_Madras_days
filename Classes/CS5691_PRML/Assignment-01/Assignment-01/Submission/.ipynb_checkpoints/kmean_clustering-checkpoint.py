"""KMeans_Lyyod
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

def kmeans(X,k,plot=False,T=1000):
    """
    k-means Clustering

    Args:
        X: [n x m] array of data, each row is a datapoint
        k: Number of clusters
        plot: Whether to plot final clustering
        T: Max number of iterations
        
    Returns:
        Numpy array of labels obtained by k-means clustering
    
    NOTE:
        z: labels
    """
    global error

    # Number of data points
    n = X.shape[0]

    # Initialization
    dist = np.zeros((k,n))
    z = np.zeros((n,))
    means = X[np.random.choice(n,size=k,replace=False),:] # Random initialization of cluster means
  

    # Main iteration for kmeans
    converged = 1
    i=0
    while i < T and converged > 0:
        
        # Update labels 
        old_labels = z.copy()
        for j in range(k):
            dist[j,:] = np.sum((X - means[j,:])**2,axis=1) # L2 Norm
        z = np.argmin(dist,axis=0)

        # Check for Convergence
        converged = np.sum(z != old_labels)

        # Update means 
        for j in range(k):
            means[j,:] = np.mean(X[z==j,:],axis=0)

        # Iterate counter
        i+=1

    # print(means)
    error.append(np.sum(dist))
    
    #Plot result (red points are labels)
    if plot:
        plt.scatter(X[:,0],X[:,1], c=z)
        plt.scatter(means[:,0],means[:,1], c='r')
        plt.title('K-means clustering')
        plt.xlabel('X_1 Features')
        plt.ylabel('X_2 Features')
        #plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
        plt.show()

    return z

data = np.genfromtxt('./Dataset.csv', delimiter=',')

#X = data
#K = 3
#kmeans_labels = kmeans(X,K,plot=True)
#print(kmeans_labels.shape)

# For 5 random Initializations with K=4:
X = data
error = []
for _ in range(5):
  num_clusters = 4 # K
  Kmeans_labels = kmeans(X, num_clusters, plot=True)
# print(error)

plt.title('K-means clustering')
plt.xlabel('Number of Initializations')
plt.ylabel('Sum of squares error (SSE)')
plt.plot(error)
#plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
plt.show()

# For K -> {2,3,4,5}
X = data
for i in range(2,6):
  num_clusters = i # K
  Kmeans_labels = kmeans(X, num_clusters, plot=True)


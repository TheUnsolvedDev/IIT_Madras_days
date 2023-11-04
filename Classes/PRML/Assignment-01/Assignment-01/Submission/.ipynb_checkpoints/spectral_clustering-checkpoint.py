"""Spectral_Clustering
Author: Jashaswimalya Acharjee
Roll: CS22E005
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

def kmeans(X,k,plot=False,T=1000):
    """
    k-means Clustering (Lyyod's)

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
        old_z = z.copy()
        for j in range(k):
            dist[j,:] = np.sum((X - means[j,:])**2,axis=1)
        z = np.argmin(dist,axis=0)

        # Check for Convergence
        converged = np.sum(z != old_z)

        # Update means 
        for j in range(k):
            means[j,:] = np.mean(X[z==j,:],axis=0)

        # Just i
        i+=1

    # Plot result (red points are cluster means)
    if plot:
        plt.scatter(X[:,0],X[:,1], c=z)
        plt.scatter(means[:,0],means[:,1], c='r')
        plt.title('K-means clustering')
        plt.xlabel('X_1 Features')
        plt.ylabel('X_2 Features')
        #plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
        plt.show()

    return z

def similarity_matrix(X, sigma):
    n = len(X)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            if i == j:
                sim_matrix[i, j] = 1   # Similarity between 2 same rows is 1
            else:
                dist = np.linalg.norm(X[i]-X[j])
                k = (-1 / (2 * (sigma**2))) * (dist**2)
                sim_matrix[i, j] = sim_matrix[j, i] = math.exp(k)
    return sim_matrix

def Laplacian_matrix(A):
    n = len(A)
    D = np.zeros((n, n))  # D is a diagonal matrix
    for i in range(n):
        k = 0
        for j in range(n):
            k += A[i, j]
        D[i, i] = k
    return D - A

def create_cluster(labels, data, total_clusters):
    clust = []  # Initilaizing the Clusters 
    m = total_clusters
    for i in range(m):   # Initilaizing each Cluster Center 
        clust.append([])
    n = len(data)
    for i in range(n):
        clust[labels[i]].append(data[i])
    return clust

def spectral_clustering(W,plot=False):
    """
    Spectral Clustering

    Args:
        W: nxn weight matrix 
        plot: Whether to scatter plot clustering

    Returns:
        Numpy array of labels obtained by spectral clustering
    """

    global K
    global sigma
    
    # K = 4

    W = similarity_matrix(W, sigma)
    L = Laplacian_matrix(W)

    # Using K-Means
    eig_vals, eig_vec = np.linalg.eig(L)
    idx = eig_vals.argsort()[0:K] # Taking top K eigenvalues.
    eig_vals = eig_vals[idx]
    v = eig_vec[:,idx]
    k_spectral = kmeans(v, K, plot=False)
    clust = create_cluster(k_spectral, v, K)

    #Plot clustering
    if plot:
      LABEL_COLOR_MAP = { 0 : 'r',
                          1 : 'k',
                          2: 'b',
                          3: 'y',
                          4: 'g',
                          5: 'm',
                        }
      label_colors = [LABEL_COLOR_MAP[l] for l in k_spectral]
      plt.figure()
      plt.scatter(X[:, 0], X[:, 1], c=label_colors)
      plt.title('Spectral Clustering')
      plt.xlabel('X_1 Features')
      plt.ylabel('X_2 Features')
      #plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
      plt.show()

    return clust, k_spectral

data = np.genfromtxt('./Dataset.csv', delimiter=',')

n=1000
X = data
K = 4
sigma = 0.15

#kmeans clustering
#kmeans_labels = kmeans(X,K,plot=True)

#Spectral clustering
final_cluster, spectral_labels = spectral_clustering(X,plot=True)

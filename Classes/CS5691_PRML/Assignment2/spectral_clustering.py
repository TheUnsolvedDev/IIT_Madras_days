import numpy as np
import matplotlib.pyplot as plt

def pairwise_distances(X, Y):
    distances = np.empty((X.shape[0], Y.shape[0]), dtype='float')
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            distances[i, j] = np.linalg.norm(X[i]-Y[j])

    return distances

def nearest_neighbor_graph(X):
    X = np.array(X)
    n_neighbors = min(int(np.sqrt(X.shape[0])), 10)
    A = pairwise_distances(X, X)
    sorted_rows_ix_by_dist = np.argsort(A, axis=1)
    nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]
    W = np.zeros(A.shape)
    for row in range(W.shape[0]):
        W[row, nearest_neighbor_index[row]] = 1
    for r in range(W.shape[0]):
        for c in range(W.shape[0]):
            if(W[r,c] == 1):
                W[c,r] = 1
    return W

def compute_laplacian(W):
    d = W.sum(axis=1)
    D = np.diag(d)
    L =  D - W
    return L

def get_eigvecs(L, k):
    eigvals, eigvecs = np.linalg.eig(L)
    ix_sorted_eig = np.argsort(eigvals)[:k]
    return eigvecs[:,ix_sorted_eig]

def k_means_pass(X, k, n_iters):
    rand_indexes = np.random.permutation(X.shape[0])[:k]
    centers = X[rand_indexes]

    for iteration in range(n_iters):
        distance_pairs = pairwise_distances(X, centers)
        labels = np.argmin(distance_pairs, axis=1)
        new_centers = [np.nan_to_num(X[labels == i].mean(axis=0)) for i in range(k)]
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers


    return centers, labels

def cluster_distance_metric(X, centers, labels):
    return sum(np.linalg.norm(X[i]-centers[labels[i]]) for i in range(len(labels)))

def k_means_clustering(X, k):
    solution_labels = None
    current_metric = None

    for pass_i in range(10):
        centers, labels = k_means_pass(X, k, 1000)
        new_metric = cluster_distance_metric(X, centers, labels)
        if current_metric is None or new_metric < current_metric:
            current_metric = new_metric
            solution_labels = labels

    return solution_labels


def spectral_clustering(X, k):
    W = nearest_neighbor_graph(X)
    L = compute_laplacian(W)
    E = get_eigvecs(L, k)
    f = k_means_clustering(E, k)
    return np.ndarray.tolist(f)

from sklearn.datasets import make_moons,make_circles

# X, y = make_moons(1000, noise=0.05)
X1, y1 = make_circles(n_samples=(250,250), factor=0.5, noise=0.02, random_state=42)
X2, y2 = make_circles(n_samples=(250,250), factor=0.5, noise=0.02, random_state=42)
X2 *= 1.5
y2 += 2
X = np.vstack([X1,X2])
y = np.concatenate([y1,y2])

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Original Data (Two Concentric Circles)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

num_clusters = max(y)+1
cluster_labels = spectral_clustering(X, num_clusters)

# Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Clustered Data using Spectral Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


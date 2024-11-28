import numpy as np
import matplotlib.pyplot as plt


class LlyodsAlgorithm:

    def compute_error(self, X, labels, centroids):
        return np.sum(((X[labels == i] - centroids[i])**2).sum() for i in range(len(centroids)))

    def initialize_centroids(self, X, k, seed=42):
        np.random.seed(seed)
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def closest_centroid(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def move_centroids(self, X, labels, k):
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])

    def lloyds_algorithm(self, X, k, seed=42, max_iters=100, tol=1e-4):
        centroids = self.initialize_centroids(X, k, seed)
        for _ in range(max_iters):
            labels = self.closest_centroid(X, centroids)
            new_centroids = self.move_centroids(X, labels, k)
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        error = self.compute_error(X, labels, centroids)
        return labels, centroids, error

def reconstruct_data(transformed_data, components, mean):
    return np.dot(transformed_data, components.T) + mean

def principal_component_analysis(data, num_components):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :num_components]
    transformed_data = np.dot(centered_data, principal_components)
    return transformed_data, principal_components, eigenvalues, mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LlyodsAlgorithm:
    def __init__(self) -> None:
        pass

    def compute_error(self, X, labels, centroids):
        return np.sum(((X[labels == i] - centroids[i])**2).sum() for i in range(len(centroids)))

    def initialize_centroids(self, X, k):
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def closest_centroid(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def move_centroids(self, X, labels, k):
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])

    def lloyds_algorithm(self, X, k, max_iters=100, tol=1e-4):
        centroids = self.initialize_centroids(X, k)
        for _ in range(max_iters):
            labels = self.closest_centroid(X, centroids)
            new_centroids = self.move_centroids(X, labels, k)
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        error = self.compute_error(X, labels, centroids)
        return labels, centroids, error


class Solution:
    def __init__(self) -> None:
        self.data = pd.read_csv('cm_dataset_2.csv')
        self.data = self.data.to_numpy()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.show()

    def question1(self):
        llyods = LlyodsAlgorithm()
        for i in range(5):
            labels, centroids, error = llyods.lloyds_algorithm(self.data, 2)
            print(f'iteration {i} error: {error} centroids: {centroids}')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels)
            plt.show()


if __name__ == '__main__':
    sol = Solution()
    sol.question1()

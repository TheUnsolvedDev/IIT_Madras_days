import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_prior = None
        self.mean = None
        self.var = None
        self.classes = None

    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_prior[idx])
            likelihood = np.sum(self._gaussian_log_likelihood(idx, x))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _gaussian_log_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = -0.5 * ((x - mean) ** 2) / var
        denominator = -0.5 * np.log(2 * np.pi * var)
        return numerator + denominator

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]=1
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.class_prior[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        return self._most_common_label(k_nearest_labels)

    def _most_common_label(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single(input_data, self.tree) for input_data in X])

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                mse_left = self._mse(y[left_indices])
                mse_right = self._mse(y[right_indices])
                weight_left = len(y[left_indices]) / n_samples
                weight_right = len(y[right_indices]) / n_samples
                mse_split = weight_left * mse_left + weight_right * mse_right

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if n_samples == 0 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return np.mean(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_subtree = self._grow_tree(
            X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(
            X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


if __name__ == '__main__':
    pass

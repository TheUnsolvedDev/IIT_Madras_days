import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from graphviz import Digraph
    
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
        n_features = X.shape[1]
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
        """
        Initialize the Decision Tree Regressor.

        Parameters:
        - max_depth: Maximum depth of the tree. If None, the tree grows until pure splits are found.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the Decision Tree model to the training data.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Target vector of shape (n_samples,)
        """
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the continuous values for the input data.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features)

        Returns:
        - predictions: Predicted values of shape (n_samples,)
        """
        return np.array([self._predict_single(input_data, self.tree) for input_data in X])

    def _mse(self, y):
        """
        Compute the Mean Squared Error (MSE) for a set of target values.

        Parameters:
        - y: Target vector of shape (n_samples,)

        Returns:
        - mse: Mean Squared Error
        """
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        """
        Find the best split for the data using Mean Squared Error (MSE).

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Target vector of shape (n_samples,)

        Returns:
        - best_feature: Index of the best feature to split on
        - best_threshold: Threshold value for the split
        """
        n_samples, n_features = X.shape
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split data based on the current threshold
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                # Compute weighted MSE of both splits
                mse_left = self._mse(y[left_indices])
                mse_right = self._mse(y[right_indices])
                weight_left = len(y[left_indices]) / n_samples
                weight_right = len(y[right_indices]) / n_samples
                mse_split = weight_left * mse_left + weight_right * mse_right

                # If this split is better, save it
                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Target vector of shape (n_samples,)
        - depth: Current depth of the tree

        Returns:
        - tree: The decision tree (a dictionary or leaf node value)
        """
        n_samples, n_features = X.shape

        # Stopping conditions
        if n_samples == 0 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        # Find the best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            return np.mean(y)

        # Split the data into left and right branches
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the tree as a dictionary
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict_single(self, x, tree):
        """
        Predict the continuous value for a single data point using the trained tree.

        Parameters:
        - x: Feature vector of a single data point
        - tree: The decision tree (a dictionary or leaf node value)

        Returns:
        - predicted value: Predicted continuous value
        """
        # If we are at a leaf node, return the leaf node value (mean of the leaf)
        if not isinstance(tree, dict):
            return tree

        # Traverse the tree based on the feature and threshold
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
        
    def export_graphviz(self, tree=None, depth=0, parent=None, dot=None):
        """
        Export the decision tree to a Graphviz .dot format.

        Parameters:
        - tree: The decision tree (default is the root)
        - depth: Current depth of the tree
        - parent: Parent node identifier
        - dot: The graphviz Dot object for building the tree
        """
        if dot is None:
            dot = Digraph()

        if tree is None:
            tree = self.tree

        # Generate a unique ID for the current node
        node_id = f"{depth}_{parent}"

        # If we are at a leaf node, add it to the dot file
        if not isinstance(tree, dict):
            dot.node(node_id, f'Predict: {tree}')
            return node_id

        # Add the decision node
        feature = tree['feature']
        threshold = tree['threshold']
        node_label = f'Feature {feature} <= {threshold:.2f}'
        dot.node(node_id, node_label)

        # Recursively add left and right subtrees
        left_child = self.export_graphviz(tree['left'], depth + 1, node_id, dot)
        right_child = self.export_graphviz(tree['right'], depth + 1, node_id, dot)

        # Connect the parent node with the children
        dot.edge(node_id, left_child, label='True')
        dot.edge(node_id, right_child, label='False')

        return node_id

    def render_tree(self, filename='tree'):
        """
        Render the tree to a .png file using Graphviz.

        Parameters:
        - filename: Name of the output file (without extension)
        """
        dot = Digraph()
        self.export_graphviz(dot=dot)
        dot.render(filename, format='png', cleanup=True)
        
if __name__ == '__main__':
    pass
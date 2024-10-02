import numpy as np


def l2_norm(x,y):
    return np.sqrt(np.square(x-y).sum())

def mse(data, labels, weights, bias):
    y = data.dot(weights) + bias
    error = y - labels
    return np.square(error).mean(axis=0)


def least_square_analytical(train_data, train_labels):
    np.random.seed(0)
    new_train_data = np.concatenate(
        [np.ones((train_data.shape[0], 1)), train_data], axis=1)
    weights = np.linalg.inv(new_train_data.T.dot(new_train_data)).dot(
        new_train_data.T).dot(train_labels)
    bias, weights = weights[0], weights[1:]
    return weights, bias


def least_square_gradient_descent(train_data, train_labels, epochs=1000, learning_rate=0.001):
    np.random.seed(0)
    weights = np.random.normal(0, 1, train_data.shape[1])*2
    bias = np.random.normal(0, 1)*2
    losses = np.zeros(epochs)
    weights_history = np.zeros((epochs, train_data.shape[1]))
    bias_history = np.zeros(epochs)
    
    for i in range(epochs):
        y = train_data.dot(weights) + bias
        error = y - train_labels
        gradient_weights = (error@train_data)
        gradient_bias = error.mean(axis=0)
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias
        losses[i] = mse(train_data, train_labels, weights, bias)
        weights_history[i] = weights
        bias_history[i] = bias
    return weights, bias, losses, (weights_history, bias_history)


def least_square_stochastic_gradient_descent(train_data, train_labels, epochs=1000, learning_rate=0.001, batch_size=32):
    np.random.seed(0)
    weights = np.random.normal(0, 1, train_data.shape[1])
    bias = np.random.normal(0, 1)
    indices = np.arange(train_data.shape[0])
    losses = np.zeros(epochs)
    weights_history = np.zeros((epochs, train_data.shape[1]))
    bias_history = np.zeros(epochs)
    
    for i in range(epochs):
        np.random.shuffle(indices)
        temp_loss = 0
        for batch in range(train_data.shape[0]//batch_size):
            new_train_data = train_data[indices[batch *
                                                batch_size:(batch+1)*batch_size]]
            new_train_labels = train_labels[indices[batch *
                                                    batch_size:(batch+1)*batch_size]]
            y = new_train_data.dot(weights) + bias
            error = y - new_train_labels
            gradient_weights = (error@new_train_data)
            gradient_bias = error.mean(axis=0)
            weights = weights - learning_rate * gradient_weights
            bias = bias - learning_rate * gradient_bias
            temp_loss += mse(new_train_data, new_train_labels, weights, bias)
        losses[i] = temp_loss / (train_data.shape[0]//batch_size)
        weights_history[i] = weights
        bias_history[i] = bias
    return weights, bias, losses, (weights_history, bias_history)


def least_square_ridge_regularization(train_data, train_labels, epochs=1000, learning_rate=0.001, lambda_=0.01):
    np.random.seed(0)
    weights = np.random.normal(0, 1, train_data.shape[1])
    bias = np.random.normal(0, 1)
    losses = np.zeros(epochs)
    for i in range(epochs):
        y = train_data.dot(weights) + bias
        error = y - train_labels
        gradient_weights = (error@train_data) + \
            2 * lambda_ * weights
        gradient_bias = error.mean(axis=0)
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias
        losses[i] = mse(train_data, train_labels, weights, bias)
    return weights, bias, losses


def polynomial_kernel(X1, X2, degree=2, coef0=1):
    return (np.dot(X1, X2.T) + coef0) ** degree


class KernelLinearRegresseion:
    def __init__(self, degree=2, coef0=1, lambda_=10):
        self.degree = degree
        self.coef0 = coef0
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.X_train = X
        polynomial_kernel_x = polynomial_kernel(
            X, X, degree=self.degree, coef0=self.coef0)
        self.weights = np.linalg.inv(polynomial_kernel_x.T.dot(
            polynomial_kernel_x) + self.lambda_ * np.eye(polynomial_kernel_x.shape[0])).dot(polynomial_kernel_x.T).dot(y)
        self.bias = np.mean(y - polynomial_kernel_x.dot(self.weights))

    def predict(self, X):
        polynomial_kernel_x = polynomial_kernel(
            X, self.X_train, degree=self.degree, coef0=self.coef0)
        return polynomial_kernel_x.dot(self.weights) + self.bias

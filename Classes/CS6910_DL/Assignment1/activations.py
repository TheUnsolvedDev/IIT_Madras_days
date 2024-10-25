import numpy as np

class Activation:
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - (np.tanh(x)**2)
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - x.max(axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

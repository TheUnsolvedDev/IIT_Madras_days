import numpy as np


def cross_entropy_loss(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def squared_error(predictions, targets):
    return np.mean(np.square(targets - predictions))
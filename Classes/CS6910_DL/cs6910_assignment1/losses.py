import numpy as np


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    '''
    accuracy() Accuracy describing the amount of correctness of the model

    :param probs: <list|np.ndarray> Probabilty vector of the prediction
    :param labels: <list|np.ndarray> True target values
    :return: <float> Accuracy value

    '''
    return np.mean(probs.argmax(axis=1) == labels.argmax(axis=1))


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12) -> float:
    '''
    cross_entropy_loss() Cross entropy loss function

    :param predictions: <list|np.ndarray> Predicted labels
    :param target: <list|np.ndarray> true labels
    :param epsilon: <float> Handling the overflow or underflow of digits
    :return: cross entropy value of the given data 
    '''
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(targets*np.log(predictions+1e-9))


def squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    '''
    cross_entropy_loss() Cross entropy loss function

    :param predictions: <list|np.ndarray> Predicted labels
    :param target: <list|np.ndarray> true labels
    :return: <float> cross entropy value of the given data 
    '''
    return np.mean(np.square(targets - predictions))


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    '''
    mean_absolute_error() Mean Absolute Error (MAE) loss function

    :param predictions: <list|np.ndarray> Predicted labels
    :param target: <list|np.ndarray> true labels
    :return: <float> Mean Absolute Error value of the given data 
    '''
    return np.mean(np.abs(targets - predictions))


def mean_squared_logarithmic_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    '''
    mean_squared_logarithmic_error() Mean Squared Logarithmic Error (MSLE) loss function

    :param predictions: <list|np.ndarray> Predicted labels
    :param target: <list|np.ndarray> true labels
    :return: <float> Mean Squared Logarithmic Error value of the given data 
    '''
    return np.mean(np.square(np.log1p(predictions) - np.log1p(targets)))


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12) -> float:
    '''
    binary_cross_entropy() Binary Cross-Entropy loss function

    :param predictions: <list|np.ndarray> Predicted labels
    :param target: <list|np.ndarray> true labels
    :param epsilon: <float> Handling the overflow or underflow of digits
    :return: <float> Binary Cross-Entropy value of the given data 
    '''
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(targets*np.log(predictions) + (1-targets)*np.log(1-predictions))

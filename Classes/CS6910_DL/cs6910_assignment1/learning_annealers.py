import numpy as np


def step_decay(initial_lr: float, decay_factor: float, epoch: int, decay_every: int) -> float:
    '''
    step_decay() Step Decay Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param decay_factor: <float> Factor by which to decay the learning rate
    :param epoch: <int> Current epoch number
    :param decay_every: <int> Decay the learning rate every decay_every epochs
    :return: <float> Updated learning rate after applying step decay
    '''
    return initial_lr * decay_factor ** (epoch // decay_every)


def exponential_decay(initial_lr: float, decay_rate: float, epoch: int) -> float:
    '''
    exponential_decay() Exponential Decay Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param decay_rate: <float> Rate of decay for the learning rate
    :param epoch: <int> Current epoch number
    :return: <float> Updated learning rate after applying exponential decay
    '''
    return initial_lr * np.exp(-decay_rate * epoch)


def inverse_decay(initial_lr: float, decay_rate: float, epoch: int) -> float:
    '''
    inverse_decay() Inverse Decay Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param decay_rate: <float> Rate of decay for the learning rate
    :param epoch: <int> Current epoch number
    :return: <float> Updated learning rate after applying inverse decay
    '''
    return initial_lr / (1 + decay_rate * epoch)


def polynomial_decay(initial_lr: float, power: float, epoch: int, total_epochs: int) -> float:
    '''
    polynomial_decay() Polynomial Decay Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param power: <float> Power of the polynomial function
    :param epoch: <int> Current epoch number
    :param total_epochs: <int> Total number of epochs
    :return: <float> Updated learning rate after applying polynomial decay
    '''
    return initial_lr * (1 - epoch / total_epochs) ** power


def piecewise_constant_decay(initial_lr: float, boundaries: list, decay_rates: list, epoch: int) -> float:
    '''
    piecewise_constant_decay() Piecewise Constant Decay Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param boundaries: <list> List of epoch boundaries for decay rate changes
    :param decay_rates: <list> List of decay rates corresponding to each boundary
    :param epoch: <int> Current epoch number
    :return: <float> Updated learning rate after applying piecewise constant decay
    '''
    for boundary, decay_rate in zip(boundaries, decay_rates):
        if epoch < boundary:
            return initial_lr * decay_rate
    return initial_lr * decay_rates[-1]


def cosine_annealing(initial_lr: float, total_epochs: int, epoch: int) -> float:
    '''
    cosine_annealing() Cosine Annealing Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param total_epochs: <int> Total number of epochs
    :param epoch: <int> Current epoch number
    :return: <float> Updated learning rate after applying cosine annealing
    '''
    return initial_lr * 0.5 * (1 + np.cos(epoch / total_epochs * np.pi))


def warm_restart(initial_lr: float, epochs_per_restart: int, restarts: int, epoch: int) -> float:
    '''
    warm_restart() Warm Restart Learning Rate

    :param initial_lr: <float> Initial learning rate
    :param epochs_per_restart: <int> Number of epochs per restart
    :param restarts: <int> Number of restarts
    :param epoch: <int> Current epoch number
    :return: <float> Updated learning rate after applying warm restart
    '''
    t_cur = epoch % epochs_per_restart
    return initial_lr * (1 - t_cur / epochs_per_restart)

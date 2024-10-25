import numpy as np
import pickle
import os


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    '''
    :param a: <list|np.ndarray> List of integers that needed to be one_hot encoded
    :param num_classes: <int>
    '''
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class BestModelCallback:
    '''
    Callback class to save the best model during training.
    '''

    def __init__(self, filepath='best_weights.pkl'):
        '''
        Initialize the callback.

        :param filepath: <str> Filepath to save the best model.
        '''
        self.filepath = filepath
        self.best_loss = np.inf
        self.best_weights = None

    def __call__(self, model_weights, current_loss):
        '''
        Callback function to save the best model.

        :param model_weights: <dict> Weights of the current model.
        :param current_loss: <float> Current loss value.
        '''
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = model_weights.copy()
            os.makedirs('weights', exist_ok=True)
            with open('weights/'+self.filepath, 'wb') as f:
                pickle.dump(self.best_weights, f)
            print(f'Saving the best model with loss: {current_loss}')

    def load_best_model(self):
        '''
        Load the best model weights.

        :return: <dict> Best model weights if available, otherwise None.
        '''
        try:
            with open('weights/'+self.filepath, 'rb') as f:
                self.best_weights = pickle.load(f)
            print(f'Loaded the best model from: {self.filepath}')
            return self.best_weights
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found.")
            return None

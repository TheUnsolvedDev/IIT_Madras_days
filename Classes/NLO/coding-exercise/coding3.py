import numpy as np
import matplotlib.pyplot as plt


class Solutions:
    x_star = np.array([3, 4])
    num_data_points = 100

    def __init__(self):
        print('''
              The solutions are presented as follows:
              ''')

    @staticmethod
    def linear_model(x, w, b):
        return x*w + b + np.random.uniform(-1,1)

    @staticmethod
    def error(y_true, y_pred):
        squared_error = np.square(y_true - y_pred)
        return squared_error.mean()
    
    @staticmethod
    def linear_regression(self):
        raise NotImplementedError
    
    def exercise1(self):
        total_steps = [100,1000]
        


if __name__ == '__main__':
    answers = Solutions()
    answers.exercise1()
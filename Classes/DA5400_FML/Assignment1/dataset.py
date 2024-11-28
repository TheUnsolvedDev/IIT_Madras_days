import numpy as np
import pandas as pd

def k_fold_split(X, y, k, shuffle=False, random_seed=None):
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))

    fold_size = len(X) // k
    for i in range(k):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]), axis=0)

        train_X, val_X = X[train_indices], X[val_indices]
        train_y, val_y = y[train_indices], y[val_indices]
        
        yield train_X, train_y, val_X, val_y

class Dataset:
    def __init__(self, train_data, train_labels, test_data, test_labels, centering = True):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.centering = centering
        
    def get_data(self, standardize = False):
        if standardize:
            mean = self.train_data.mean(axis=0)
            std = self.train_data.std(axis=0)
            self.train_data = (self.train_data - mean) / std
            self.test_data = (self.test_data - mean) / std
        return self.train_data, self.train_labels, self.test_data, self.test_labels
    
def generate_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_data = np.array(train)[:, :-1]
    train_labels = np.array(train)[:, -1]
    test_data = np.array(test)[:, :-1]
    test_labels = np.array(test)[:, -1]
    return Dataset(train_data, train_labels, test_data, test_labels)

if __name__ == '__main__':
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    
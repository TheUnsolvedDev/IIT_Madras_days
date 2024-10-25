import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class CreateDataset:
    def __init__(self, num_features, num_classes=2, num_data=1000, seperation=2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.seperation = seperation

        self.X1 = np.random.randn(num_data//2, num_features) + seperation
        self.X2 = np.random.randn(num_data//2, num_features) - seperation
        self.X = np.concatenate((self.X1, self.X2), axis=0)
        self.y = np.concatenate((np.ones(num_data//2), np.zeros(num_data//2)), axis=0).reshape(-1,1)

        os.makedirs('dataset', exist_ok=True)
        self.write_data()

    def get_data(self):
        return self.X, self.y
    
    def write_data(self):
        dataset = np.hstack((self.X, self.y))
        np.random.shuffle(dataset)
        train_data = dataset[:int(0.8*len(dataset))]
        test_data = dataset[int(0.8*len(dataset)):]
        pd.DataFrame(train_data).to_csv('dataset/train.csv', index=False)
        pd.DataFrame(test_data).to_csv('dataset/test.csv', index=False)
        
def dataset():
    if not os.path.exists('dataset'):
        dataset = CreateDataset(num_features=2)
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    train = np.array(train)
    test = np.array(test)
    
    train_data,train_labels = train[:,:-1],train[:,-1]
    test_data,test_labels = test[:,:-1],test[:,-1]
    return train_data,train_labels,test_data,test_labels
        

if __name__ == '__main__':
    dataset = CreateDataset(num_features=2)
    X, y = dataset.get_data()
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    print(X.shape, y.shape)
    print(np.hstack((X, y)).shape)

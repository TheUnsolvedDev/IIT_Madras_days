import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class CreateLinearSeparatedDataset:
    def __init__(self, num_features, num_classes=2, num_data=1000, seperation=2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.seperation = seperation

        self.X1 = np.random.randn(num_data//2, num_features) + seperation
        self.X2 = np.random.randn(num_data//2, num_features) - seperation
        self.X = np.concatenate((self.X1, self.X2), axis=0)
        self.y = np.concatenate(
            (np.ones(num_data//2), np.zeros(num_data//2)), axis=0).reshape(-1, 1)

        os.makedirs('train', exist_ok=True)
        os.makedirs('test', exist_ok=True)
        self.write_data()

    def get_data(self):
        return self.X, self.y

    def write_data(self):
        dataset = np.hstack((self.X, self.y))
        np.random.shuffle(dataset)
        train_data = dataset[:int(0.8*len(dataset))]
        test_data = dataset[int(0.8*len(dataset)):]
        pd.DataFrame(train_data).to_csv(
            'train/emial1.txt', index=False)
        pd.DataFrame(test_data).to_csv(
            'test/emial1.txt', index=False)


class CreateNonLinearSeparatedDataset:
    def __init__(self, num_features, num_classes=2, num_data=10000):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_data = num_data

        noise, factor = 0.1, 0.5
        num_samples_per_circle = num_data // 2
        theta = np.linspace(0, 2 * np.pi, num_samples_per_circle)

        r_outer = 1
        x_outer = r_outer * \
            np.cos(theta) + np.random.normal(0, noise, num_samples_per_circle)
        y_outer = r_outer * \
            np.sin(theta) + np.random.normal(0, noise, num_samples_per_circle)

        r_inner = factor
        x_inner = r_inner * \
            np.cos(theta) + np.random.normal(0, noise, num_samples_per_circle)
        y_inner = r_inner * \
            np.sin(theta) + np.random.normal(0, noise, num_samples_per_circle)

        self.X = np.vstack((np.column_stack((x_inner, y_inner)),
                           np.column_stack((x_outer, y_outer))))
        self.y = np.hstack((np.zeros(num_samples_per_circle),
                           np.ones(num_samples_per_circle))).reshape(-1, 1)

        os.makedirs('train', exist_ok=True)
        os.makedirs('test', exist_ok=True)
        self.write_data()

    def get_data(self):
        return self.X, self.y

    def write_data(self):
        dataset = np.hstack((self.X, self.y))
        np.random.shuffle(dataset)
        train_data = dataset[:int(0.8*len(dataset))]
        test_data = dataset[int(0.8*len(dataset)):]
        pd.DataFrame(train_data).to_csv(
            'train/email2.txt', index=False)
        pd.DataFrame(test_data).to_csv(
            'test/email2.txt', index=False)


def dataset(type='linear'):
    if type == 'linear':
        if not os.path.exists('dataset_l'):
            dataset = CreateLinearSeparatedDataset(
                num_features=10, seperation=1)
        train = pd.read_csv('train/emial1.txt')
        test = pd.read_csv('test/emial1.txt')
    if type == 'nonlinear':
        if not os.path.exists('dataset_nl'):
            dataset = CreateNonLinearSeparatedDataset(num_features=10)
        train = pd.read_csv('train/email2.txt')
        test = pd.read_csv('test/email2.txt')
    train = np.array(train)
    test = np.array(test)

    train_data, train_labels = train[:, :-1], train[:, -1]
    test_data, test_labels = test[:, :-1], test[:, -1]
    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    dataset = CreateNonLinearSeparatedDataset(num_features=2)
    X, y = dataset.get_data()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    print(X.shape, y.shape)
    print(np.hstack((X, y)).shape)

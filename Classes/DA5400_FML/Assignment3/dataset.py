import pandas as pd
import numpy as np
import struct
from array import array


def read_images(images_filepath, labels_filepath):
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(
                'Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(
                'Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img
    return images,labels


class Dataset1:
    def __init__(self) -> None:
        self.train = read_images('dataset/train-images.idx3-ubyte',
                                 'dataset/train-labels.idx1-ubyte')
        self.test = read_images('dataset/t10k-images.idx3-ubyte',
                                'dataset/t10k-labels.idx1-ubyte')
        
    def get_data(self):
        images = []
        class_counter = {i:0 for i in range(10)}
        for i in range(len(self.train[0])):
            if class_counter[self.train[1][i]] < 100:
                images.append(self.train[0][i])
                class_counter[self.train[1][i]] += 1
        return np.array(images,dtype=np.uint8).reshape(-1,28,28)


class Dataset2:
    def __init__(self, file_path='dataset/cm_dataset_2.csv') -> None:
        self.data = pd.read_csv(file_path)
        self.data_numpy = self.data.to_numpy()

    def standardize(self, data):
        return (data - data.mean()) / data.std()


if __name__ == '__main__':
    dataset = Dataset1()
    # print(dataset.data_numpy.shape)

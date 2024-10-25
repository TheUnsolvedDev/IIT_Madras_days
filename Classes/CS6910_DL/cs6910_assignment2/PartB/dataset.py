import torchvision
import torch
import tqdm
import numpy as np
import os
from config import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Dataset class
class Dataset:
    def __init__(self, train_path: str = 'inaturalist_12K/train', test_path: str = 'inaturalist_12K/val', input_shape: list = IMAGE_SHAPE, batch_size: int = BATCH_SIZE, transformation=True):
        """
        Initialize Dataset class.

        Parameters:
        - train_path (str): Path to the training dataset.
        - test_path (str): Path to the test dataset.
        - input_shape (list): Input shape of images.
        - batch_size (int): Batch size for DataLoader.
        - transformation (bool): Whether to apply data transformations.
        """
        self.train_path = train_path
        self.test_path = test_path

        if transformation == False:
            # If transformation is False, apply only resize and ToTensor transformations
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_shape),
                torchvision.transforms.ToTensor(),
            ])
        else:
            # If transformation is True, apply additional transformations including RandomHorizontalFlip and normalization
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_shape),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Load the full dataset
        self.full_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transform)
        self.batch_size = batch_size

    def get_data(self):
        """
        Prepare and return DataLoader for training, validation, and test datasets.
        """
        classes = self.full_dataset.classes
        print('Preparing Dataset please wait!!')
        class_to_data = {c: [] for c in classes}
        for data, label in tqdm.tqdm(self.full_dataset):
            # Organize data by class
            class_to_data[classes[label]].append((data, label))

        train_data = []
        val_data = []
        for c, data in (class_to_data.items()):
            # Split data into training and validation sets
            train_size = int(len(data) * 0.8)
            train_data.extend(data[:train_size])
            val_data.extend(data[train_size:])
        
        # Create DataLoaders for training, validation, and test sets
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=True)
        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    dataset = Dataset()
    train, val, test = dataset.get_data()
    # Iterate through validation data to print shapes
    for imgs, lbls in val:
        print(imgs.shape, lbls)

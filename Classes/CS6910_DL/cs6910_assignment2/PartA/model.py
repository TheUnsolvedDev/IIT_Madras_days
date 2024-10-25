import torch
import torchsummary
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from config import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(torch.nn.Module):
    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 input_shape: list = IMAGE_SHAPE,
                 in_channels: int = NUM_CHANNELS,
                 conv_filters: list = [32, 64, 128, 256, 512],
                 conv_kernel_sizes: list = [7, 7, 5, 3, 3],
                 conv_strides: list = [1, 1, 1, 1, 1],
                 pool_kernel_sizes: list = [2, 2, 2, 2, 2],
                 pool_strides: list = [2, 2, 2, 2, 2],
                 dense_units: int = 256,
                 dropout_rate: float = 0.5,
                 batch_normalization: bool = True,
                 activation: torch.nn.Module = torch.nn.ReLU()) -> None:
        """
        Initialize CNN model.

        Parameters:
        - num_classes (int): Number of output classes.
        - input_shape (list): Input shape of images.
        - in_channels (int): Number of input channels.
        - conv_filters (list): List of number of filters for each convolutional layer.
        - conv_kernel_sizes (list): List of kernel sizes for each convolutional layer.
        - conv_strides (list): List of strides for each convolutional layer.
        - pool_kernel_sizes (list): List of kernel sizes for each pooling layer.
        - pool_strides (list): List of strides for each pooling layer.
        - dense_units (int): Number of units in the dense layer.
        - dropout_rate (float): Dropout rate.
        - batch_normalization (bool): Whether to use batch normalization.
        - activation (torch.nn.Module): Activation function.
        """
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.batch_normalization = batch_normalization
        shapes, img_size = [], self.input_shape[0]

        self.conv_blocks = torch.nn.ModuleList()
        in_channels = in_channels
        self.activations = [activation for i in range(len(conv_filters) + 1)]
        for i in range(len(conv_filters)):
            conv_layer = torch.nn.Conv2d(in_channels, conv_filters[i], kernel_size=conv_kernel_sizes[i],
                                         stride=conv_strides[i], padding=conv_kernel_sizes[i] // 2)
            self.conv_blocks.append(conv_layer)
            if self.batch_normalization:
                self.conv_blocks.append(torch.nn.BatchNorm2d(
                    conv_filters[i]))
            self.conv_blocks.append(self.activations[i])
            in_channels = conv_filters[i]
            shapes.append(self._shape_after_conv(
                img_size, in_channels, conv_kernel_sizes[i], conv_strides[i], pool_kernel_sizes[i], pool_strides[i]))
            img_size = shapes[-1][-1]
            pool_layer = torch.nn.MaxPool2d(
                kernel_size=pool_kernel_sizes[i], stride=pool_strides[i])
            self.conv_blocks.append(pool_layer)

        linear_conversion = np.prod(shapes[-1])
        self.fc = torch.nn.Linear(linear_conversion, dense_units)
        self.output = torch.nn.Linear(dense_units, num_classes)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def _shape_after_conv(self, input_size: int, in_channels: int, kernel_size: int, stride: int, pool_kernel_size: int, pool_stride: int):
        """
        Calculate the shape after convolution and pooling.

        Parameters:
        - input_size (int): Input size.
        - in_channels (int): Number of input channels.
        - kernel_size (int): Kernel size.
        - stride (int): Stride.
        - pool_kernel_size (int): Pooling kernel size.
        - pool_stride (int): Pooling stride.

        Returns:
        - tuple: Shape after convolution and pooling.
        """
        output_size_conv = (input_size - kernel_size) // stride + 2
        output_size_pool = (output_size_conv -
                            pool_kernel_size) // pool_stride + 2
        return (in_channels, output_size_pool, output_size_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        for layer in self.conv_blocks:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.activations[-1](x)
        x = self.dropout1(x)
        x = self.output(x)
        return x


def training(model, datasets):
    """
    Training function.

    Parameters:
    - model: The model to be trained.
    - datasets: Tuple containing training and validation datasets.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    train, val = datasets
    torchsummary.summary(model, (NUM_CHANNELS, *IMAGE_SHAPE))
    loss_thresh = -np.inf
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = []
        train_running_accuracy = []

        pbar = tqdm.tqdm(
            train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')
        for images, labels in pbar:
            train_images, train_labels = images.to(device), labels.to(device)
            train_outputs = model(train_images)
            train_loss = criterion(train_outputs, train_labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_running_loss.append(train_loss.item())
            train_predicted = torch.argmax(train_outputs.data, 1)
            train_total = train_labels.shape[0]
            train_correct = (train_predicted == train_labels).sum().item()
            train_accuracy = train_correct/train_total
            train_running_accuracy.append(train_accuracy)

            pbar.set_postfix({
                'loss': np.mean(train_running_loss),
                'accuracy': np.mean(train_running_accuracy)
            })
        epoch_loss = train_loss.item()
        epoch_accuracy = train_accuracy
        print(
            f'Training Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')
        wandb.log({'train_accuracy': epoch_accuracy, 'train_loss': epoch_loss})
        writer.add_scalar('Train/Loss', np.mean(train_running_loss), epoch)
        writer.add_scalar('Train/Accuracy',
                          np.mean(train_running_accuracy), epoch)

        model.eval()
        val_running_loss = []
        val_running_accuracy = []

        pbar = tqdm.tqdm(
            val, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')
        with torch.no_grad():
            for images, labels in pbar:
                val_images, val_labels = images.to(device), labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss.append(val_loss.item())
                val_predicted = torch.argmax(val_outputs.data, 1)
                val_total = val_labels.shape[0]
                val_correct = (val_predicted == val_labels).sum().item()
                val_accuracy = val_correct/val_total
                val_running_accuracy.append(val_accuracy)

                pbar.set_postfix({
                    'loss': np.mean(val_running_loss),
                    'accuracy': np.mean(val_running_accuracy)
                })
            epoch_loss = val_loss.item()
            epoch_accuracy = val_accuracy
            print(
                f'Validation Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')
            wandb.log({'val_accuracy': epoch_accuracy,
                      'val_loss': epoch_loss})
            writer.add_scalar('Validation/Loss',
                              np.mean(val_running_loss), epoch)
            writer.add_scalar('Validation/Accuracy',
                              np.mean(val_running_accuracy), epoch)

        if np.mean(val_running_loss) > loss_thresh:
            loss_thresh = np.mean(val_running_loss)
            torch.save(model.state_dict(), './cnn_model.pth')
        print()

    print('Training finished!')


if __name__ == '__main__':
    model = CNN().cuda()
    torchsummary.summary(model, (NUM_CHANNELS, *IMAGE_SHAPE))

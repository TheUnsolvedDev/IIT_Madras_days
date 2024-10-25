import torch
import torchsummary
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt

# Import custom modules
from config import *
from model import *
from dataset import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_image(tensor):
    """
        Convert tensor to image for matplotlib

        Parameters:
        - tensor (torch.Tensor): Input tensor.

        Returns:
        - np.ndarray: Output array.
        """
    # Assuming tensor is of shape (C, H, W) where C=3
    tensor = tensor.permute(1, 2, 0)  # Reorder dimensions for visualization
    return tensor.numpy()


def test_model():
    """
    Fix the model before using it according to the save model
    """
    model = CNN(conv_filters=[128*(2**i)
                for i in range(5)], dense_units=256, activation=torch.nn.Mish()).to(device)
    model.load_state_dict(torch.load('cnn_model.pth'))
    torchsummary.summary(model, (NUM_CHANNELS, *IMAGE_SHAPE))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SHAPE),
        torchvision.transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.ImageFolder(
        'inaturalist_12K/val', transform=transform)
    test = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    classes = test_dataset.classes

    model.eval()
    test_running_accuracy = []

    batch_images, batch_labels, batch_predictions = [], [], []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test):
            test_images, test_labels = images.to(device), labels.to(device)
            test_outputs = model(test_images)

            test_predicted = torch.argmax(test_outputs.data, 1)
            test_total = test_labels.shape[0]
            test_correct = (test_predicted == test_labels).sum().item()
            test_accuracy = test_correct/test_total
            test_running_accuracy.append(test_accuracy)

            if len(batch_images) < 30:
                for image in images:
                    batch_images.append(image)
                for label in labels:
                    batch_labels.append(label)
                for pred in test_predicted:
                    batch_predictions.append(pred)

    batch_predictions = batch_predictions[:30]
    batch_images = batch_images[:30]
    batch_labels = batch_labels[:30]

    print('Test accuracy:', np.mean(test_running_accuracy))

    with wandb.init(project='CS23E001_DL_2') as run:
        fig, ax = plt.subplots(3, 10, figsize=(60, 18))
        plt.tight_layout()
        for x in range(10):
            for y in range(3):
                ax[y][x].imshow(tensor_to_image(batch_images[x*3+y]))
                ax[y][x].axis('off')
                if classes[batch_labels[x*3+y]] == classes[batch_predictions[x*3+y]]:
                    color = 'green'
                else:
                    color = 'red'
                ax[y][x].set_title(
                    f'True label: {classes[batch_labels[x*3+y]]} Predicted: {classes[batch_predictions[x*3+y]]}', color=color)
        wandb.log({"plot": fig})


if __name__ == "__main__":
    test_model()

import torch
import torchsummary
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt

# Import custom modules
from config import *
from dataset import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_image(tensor):
    # Assuming tensor is of shape (C, H, W) where C=3
    tensor = tensor.permute(1, 2, 0)  # Reorder dimensions for visualization
    return tensor.numpy()


def build_model(pretrained: bool = True, fine_tune: bool = True, num_classes: int = 10) -> torch.nn.Module:
    """
    Build an EfficientNet-V2 model for a specific task.

    Parameters:
    - pretrained (bool): Whether to load pre-trained weights.
    - fine_tune (bool): Whether to fine-tune all layers or freeze hidden layers.
    - num_classes (int): Number of output classes.

    Returns:
    - torch.nn.Module: Built EfficientNet-V2 model.
    """
    if pretrained:
        model = torchvision.models.efficientnet_v2_m(
            weights='EfficientNet_V2_M_Weights.IMAGENET1K_V1')
        print('[INFO]: Loading pre-trained weights')
    else:
        model = torchvision.models.efficientnet_v2_m()
        print('[INFO]: Not loading pre-trained weights')

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing cnn layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=512)

    # Add dropout and final classification layer
    model = torch.nn.Sequential(
        model,
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=512, out_features=num_classes)
    )
    return model


def test_model():
    model = build_model().cuda()
    model.load_state_dict(torch.load('partb_cnn_model.pth'))
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

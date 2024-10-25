import numpy as np
import torch
import wandb
import os
import argparse
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torchsummary

# Import custom modules
from config import *
from dataset import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    if not fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing hidden layers...')
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


def train_efficientnet(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    dataset = Dataset(batch_size=32)
    train, val, test = dataset.get_data()
    loss_thresh = -np.inf
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = []
        train_running_accuracy = []

        pbar = tqdm.tqdm(
            train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')
        for images, labels in pbar:
            train_images, train_labels = images.to(
                device), labels.to(device)
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
        wandb.log({'train_accuracy': epoch_accuracy,
                   'train_loss': epoch_loss})
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
                val_images, val_labels = images.to(
                    device), labels.to(device)
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
            torch.save(model.state_dict(), './partb_cnn_model.pth')
        print()

    print('Training finished!')


def main():
    parser = argparse.ArgumentParser(
        description='Finetuning a pretrained model')
    parser.add_argument('-ft', '--fine_tune', type=bool,
                        default=False, help='Finetuning Enabled or not')
    parser.add_argument('-pt', '--pre_train', type=bool,
                        default=False, help='Pretrained or not')
    args = parser.parse_args()
    print(f'Part B Experiment Fine tuning {args.fine_tune} Pretrained {args.pre_train}')
    with wandb.init(project='CS23E001_DL_2') as run:
        run.name = f'Part B Experiment Fine tuning {args.fine_tune} Pretrained {args.pre_train}'
        print(run.name)
        model = build_model(pretrained=args.pre_train,
                            fine_tune=args.fine_tune).cuda()
        torchsummary.summary(model, (NUM_CHANNELS, *IMAGE_SHAPE))
        train_efficientnet(model)


if __name__ == "__main__":
    main()

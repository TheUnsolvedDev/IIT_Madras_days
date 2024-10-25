import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training(model, datasets):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train, val = datasets
    torchsummary.summary(model, (NUM_CHANNELS, *IMAGE_SHAPE))
    loss_thresh = -np.inf
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = []
        correct = 0
        total = 0
        pbar = tqdm.tqdm(
            train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct/total

            pbar.set_postfix({
                'loss': np.mean(running_loss),
                'accuracy': accuracy
            })
        epoch_loss = np.sum(loss.item())/len(train)
        print(
            f'Training Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}')
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)

        model.eval()
        running_loss = []
        correct = 0
        total = 0
        pbar = tqdm.tqdm(
            val, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='batch')
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = correct/total

                pbar.set_postfix({
                    'loss': np.mean(running_loss),
                    'accuracy': accuracy
                })
            epoch_loss = np.sum(loss.item())/len(test)
            print(
                f'Testing Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}')

        if np.mean(running_loss) > loss_thresh:
            loss_thresh = np.mean(running_loss)
            torch.save(model.state_dict(), './cnn_model.pth')
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    print('Training finished!')


if __name__ == '__main__':
    model = CNN().to(device)
    data = Dataset()
    train, val, test = data.get_data()
    training(model, [train, val])

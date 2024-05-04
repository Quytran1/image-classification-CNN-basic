import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import classification_report

from dataset import get_loaders
from models.CNN import Net
from config import configuration


def train(params):
    # get hyperparameters
    batch_size = params["batch_size"]
    lr = params["lr"]
    epochs = params["epochs"]

    # Get data
    train_loader, test_loader = get_loaders(batch_size)

    # define model
    model = Net()
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    
    # define criterion
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # training
    best_loss = float('inf')
    best_acc = 0.0
    training_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(tqdm.tqdm(train_loader, desc="Training", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            # get data
            imgs, labels = data

            # zero optimizer
            optimizer.zero_grad()

            # forward pass
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).sum().item() / labels.size(0)

            # backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update running variables
            running_loss += loss.item()
            running_acc += acc

        running_acc /= len(train_loader)    
        running_loss /= len(train_loader)
        training_loss.append(running_loss)
        print("Epoch {}/{} | Loss: {:.4f} | Accuracy: {:.4f}".format(epoch + 1, epochs, running_loss, running_acc))
        
        # save model   
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss,
            "accuracy": running_acc
        }, f'logs/checkpoints/checkpoint_{epoch+1}.ckpt')

        if best_loss > running_loss:
            best_loss = running_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
                "accuracy": running_acc
            }, 'logs/checkpoints/best_loss_checkpoint.ckpt')

    # plot loss
    training_loss = np.array(training_loss)
    plt.plot(training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("logs/history/loss_history")


def test(params):
    # get hyperparameters
    batch_size = params["batch_size"]

    # Get data
    train_loader, test_loader = get_loaders(batch_size)

    # define model
    model = Net()
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    chetkpoint = torch.load("logs/checkpoints/best_loss_checkpoint.ckpt")
    model.load_state_dict(chetkpoint["model_state_dict"])
    model.eval()

    running_acc = 0.0
    y_true = []
    y_pred = []
    for i, data in enumerate(tqdm.tqdm(test_loader, desc="Testing", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        # get data
        imgs, labels = data

        # forward pass
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).sum().item() / labels.size(0)

        # update running variables
        running_acc += acc
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

    running_acc /= len(test_loader)    
    print("Accuracy: {:.2f}".format(running_acc))
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    train(params=configuration)
    
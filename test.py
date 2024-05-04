import os

import torch
import tqdm
from sklearn.metrics import classification_report

from dataset import get_loaders
from models.CNN import Net
from config import configuration


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
    test(params=configuration)

import torch
import torchvision #cung cấp các công cụ để làm việc với dữ liệu
import torchvision.transforms as transforms # thực hiện các biến đổi dữ liệu ảnh
from torch.utils.data import Dataset, DataLoader

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

def get_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #Tải và lưu vào đối tượng trainset bộ dữ liệu train cifar10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    #Tải và lưu vào đối tượng testset bộ dữ liệu test cifar10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_cifar10(args, data_dir='./data/cifar/'):
    """Returning cifar dataloder."""
    transform = transforms.Compose([transforms.Resize(32), #3x32x32 images.
                                    transforms.ToTensor()])
    data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader


def get_mnist(args, data_dir='./data/mnist/'):
    """Returning mnist dataloder."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader

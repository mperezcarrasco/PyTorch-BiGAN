import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_cifar10(args, data_dir='./data/cifar/'):
    # Normalizing data in range [-1, 1] 
    transform = transforms.Compose([transforms.Scale(32), #3x32x32 images.
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dataset.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
import numpy as np
import argparse 
import torch

from train import TrainerBiGAN
from preprocess import get_cifar10, get_mnist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument('--lr_adam', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_rmsprop', type=float, default=1e-4,
                        help='learning rate RMSprop if WGAN is True.')
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of the latent variable z')
    parser.add_argument('--wasserstein', type=bool, default=False,
                        help='If WGAN.')
    parser.add_argument('--clamp', type=float, default=1e-2,
                        help='Clipping gradients for WGAN.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                        help='Clipping gradients for WGAN.')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data = get_cifar10(args)
    elif args.dataset == 'mnist':
        data = get_mnist(args)

    bigan = TrainerBiGAN(args, data, device)
    bigan.train()


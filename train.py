import torch
from torch import optim
import torch.nn.functional as F
import torchvision.utils as vutils

import numpy as np
from barbar import Bar

from model import Generator, Encoder, Discriminator
from utils.utils import weights_init_normal


class TrainerBiGAN:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader = data
        self.device = device
    

    def train(self):
        """Training the BiGAN"""
        self.G = Generator(self.args.latent_dim).to(self.device)
        self.E = Encoder(self.args.latent_dim).to(self.device)
        self.D = Discriminator(self.args.latent_dim, self.args.wasserstein).to(self.device)
        
        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        
        if args.wasserstein:
            optimizer_ge = optim.RMSprop(list(self.G.parameters()) + 
                                      list(self.E.parameters()),, lr=self.args.lr_rmsprop)
            optimizer_d = optim.RMSprop(self.D.parameters(), lr=self.args.lr_rmsprop)
        else:
            optimizer_ge = optim.Adam(list(self.G.parameters()) + 
                                      list(self.E.parameters()),, lr=self.args.lr_adam)
            optimizer_d = optim.Adam(self.D.parameters(), lr=self.args.lr_adam)

        fixed_z = Variable(torch.randn((10, self.args.latent_dim, 1, 1)),
                           requires_grad=False).to(self.device)
        criterion = nn.BCELoss()
        for epoch in range(self.args.num_epochs):
            g_losses = 0
            d_losses = 0
            for x, _ in Bar(self.train_loader):
                #Defining labels
                y_true = Variable(torch.ones((x.size(0), 1)).to(self.device))
                y_fake = Variable(torch.zeros((x.size(0), 1)).to(self.device))

                #Noise for improving training.
                noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs),
                                  requires_grad=False).to(self.device)
                noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs),
                                  requires_grad=False).to(self.device)

                #Cleaning gradients.
                optimizer_d.zero_grad()
                optimizer_ge.zero_grad()

                #Generator:
                z_fake = Variable(torch.randn((x.size(0), self.args.latent_dim, 1, 1)).to(self.device),
                                  requires_grad=False)
                x_fake = self.G(z_fake)

                #Encoder:
                x_true = x.float().to(self.device)
                z_true = self.E(x_true)
                
                #Discriminator
                out_true = self.D(x_true + noise1, z_true)
                out_fake = self.D(x_fake + noise2, z_fake)

                #Losses
                if self.args.wasserstein:
                    loss_d = - torch.mean(out_true) + torch.mean(out_fake)
                    loss_ge = - torch.mean(out_fake) + torch.mean(out_true)
                else:
                    loss_d = criterion(out_true, y_true) + criterion(out_fake, y_fake)
                    loss_ge = criterion(out_fake, y_true) + criterion(out_true, y_fake)

                #Computing gradients and backpropagate.
                loss_d.backward(retain_variables=True)
                optimizer_d.step()

                loss_ge.backward()
                optimizer_ge.step()


                if self.args.wasserstein:
                    for p in self.D.parameters():
                        p.data.clamp_(-self.args.clamp, self.args.clamp)

            if epoch % 50 == 0:
                vutils.save_image(self.G(fixed_z).data, './images/{}_fake.png'.format(epoch))

            print("Training... Epoch: {}, Discrimiantor Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), g_losses/len(self.train_loader)
            ))
                

        


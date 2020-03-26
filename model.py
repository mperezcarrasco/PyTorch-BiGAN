import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, z_dim=200, wasserstein=False):
        super(Discriminator, self).__init__()
        self.wass = wasserstein

        # Inference over x
        self.conv1x = nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False)
        self.conv2x = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)
        self.bn2x = nn.BatchNorm2d(64)

        # Inference over z
        self.nn1z = nn.Linear(z_dim, 512, bias=False)

        # Joint inference
        self.nn1xz = nn.Linear(7*7*64 + 512, 1024)
        self.nn2xz = nn.Linear(1024, 1)

    def inf_x(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1x(x), negative_slope=0.1), 0.5)
        x = F.dropout2d(F.leaky_relu(self.bn2x(self.conv2x(x)), negative_slope=0.1), 0.5)
        return x

    def inf_z(self, z):
        z = F.dropout(F.leaky_relu(self.nn1z(z), negative_slope=0.1), 0.5)
        return z

    def inf_xz(self, xz):
        xz = F.dropout(F.leaky_relu(self.nn1xz(xz), negative_slope=0.1), 0.5)
        xz = self.nn2xz(xz)
        return xz

    def forward(self, x, z):
        x = self.inf_x(x)
        z = z.view(z.size(0),-1)
        z = self.inf_z(z)
        xz = torch.cat((x.view(x.size(0),-1),z), dim=1)
        out = self.inf_xz(xz)
        if self.wass:
            return out
        else:
            return torch.sigmoid(out)


class Generator(nn.Module):
    def __init__(self, z_dim=32):
        super(Generator, self).__init__()
        self.output_bias = nn.Parameter(torch.zeros(1, 28, 28), requires_grad=True)
        
        self.nn1 = nn.Linear(z_dim, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.nn2 = nn.Linear(1024, 7*7*128, bias=False)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False)

    def forward(self, z):
        z = z.view(z.size(0),-1)
        z = F.relu(self.bn1(self.nn1(z)))
        z = F.relu(self.bn2(self.nn2(z)))
        z = z.view(z.size(0), 128, 7, 7)
        z = F.relu(self.bn3(self.deconv3(z)))
        z = F.relu(self.deconv4(z)) + self.output_bias
        return torch.tanh(z)


class Encoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.nn4 = nn.Linear(128*5*5, z_dim*2)

    def reparameterize(self, z):
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = x.view(x.size(0),-1)
        z = self.reparameterize(self.nn4(x))
        return z.view(x.size(0), self.z_dim, 1, 1)

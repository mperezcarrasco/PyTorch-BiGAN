import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, z_dim=32, wasserstein=False):
        super(Discriminator, self).__init__()
        self.wass = wasserstein

        # Inference over x
        self.conv1x = nn.Conv2d(3, 32, 5, stride=1, bias=False)
        self.conv2x = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.bn2x = nn.BatchNorm2d(64)
        self.conv3x = nn.Conv2d(64, 128, 4, stride=1, bias=False)
        self.bn3x = nn.BatchNorm2d(128)
        self.conv4x = nn.Conv2d(128, 256, 4, stride=2, bias=False)
        self.bn4x = nn.BatchNorm2d(256)
        self.conv5x = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.bn5x = nn.BatchNorm2d(512)

        # Inference over z
        self.conv1z = nn.Conv2d(z_dim, 512, 1, stride=1, bias=False)
        self.conv2z = nn.Conv2d(512, 512, 1, stride=1, bias=False)

        # Joint inference
        self.conv1xz = nn.Conv2d(1024, 1024, 1, stride=1, bias=False)
        self.conv2xz = nn.Conv2d(1024, 1024, 1, stride=1, bias=False)
        self.conv3xz = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def inf_x(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1x(x), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn2x(self.conv2x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn3x(self.conv3x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn4x(self.conv4x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn5x(self.conv5x(x)), negative_slope=0.1), 0.2)
        return x

    def inf_z(self, z):
        z = F.dropout2d(F.leaky_relu(self.conv1z(z), negative_slope=0.1), 0.2)
        z = F.dropout2d(F.leaky_relu(self.conv1z(z), negative_slope=0.1), 0.2)
        return z

    def inf_xz(self, xz):
        xz = F.dropout(F.leaky_relu(self.conv1xz(xz), negative_slope=0.1), 0.2)
        xz = F.dropout(F.leaky_relu(self.conv2xz(xz), negative_slope=0.1), 0.2)
        return self.conv3xz(xz)

    def forward(self, x, z):
        x = self.inf_x(x)
        z = self.inf_z(z)
        xz = torch.cat((x,z), dim=1)
        out = self.inf_xz(xz)
        if self.was:
            return out
        else: 
            return torch.sigmoid(out)


class Generator(nn.Module):
    def __init__(self, z_dim=32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
        self.deconv1 = nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv6 = nn.Conv2d(32, 3, 1, stride=1, bias=True)
   
    def forward(self, z):
        z = F.leaky_relu(self.bn1(self.deconv1(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn2(self.deconv2(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn3(self.deconv3(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn4(self.deconv4(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn5(self.deconv5(z)), negative_slope=0.1)
        return torch.tanh(self.deconv6(z) + self.output_bias)


class Encoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512) 
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512) 
        self.bn7 = nn.Conv2d(512, z_dim*2, 1, stride=1, bias=True)
    
    def reparameterize(self, z):
        z = z.view(z.size(0), -1)
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.1)
        z = self.reparameterize(self.conv6(x))
        return z.view(x.size(0), self.z_dim, 1, 1)
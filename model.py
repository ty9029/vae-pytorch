import torch
import torch.nn as nn
import math


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTranspose2d, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.convt(x)
        out = self.relu(out)

        return out


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim):
        super(Encoder, self).__init__()
        down_size = 4
        num_layer = int(math.log2(image_size) - math.log2(down_size))

        self.conv = []
        self.conv.append(Conv2d(image_channels, 16, 3, 1, 1))
        for i in range(num_layer):
            num_channels = 2 ** min((4 + i), 9)
            self.conv.append(Conv2d(num_channels, num_channels * 2, 3, 2, 1))

        self.conv = nn.Sequential(*self.conv)

        num_channels = 2 ** min((4 + num_layer), 9)
        self.fc_mean = nn.Linear(down_size ** 2 * num_channels, latent_dim)
        self.fc_logvar = nn.Linear(down_size ** 2 * num_channels, latent_dim)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim):
        super(Decoder, self).__init__()
        self.down_size = 4
        num_layer = int(math.log2(image_size) - math.log2(self.down_size))

        self.base_channels = 2 ** min((4 + num_layer), 9)
        self.fc = nn.Linear(latent_dim, self.down_size ** 2 * self.base_channels)

        self.conv = []
        for i in reversed(range(num_layer)):
            num_channels = 2 ** min((4 + i), 9)
            self.conv.append(ConvTranspose2d(num_channels * 2, num_channels, 4, 2, 1))

        self.conv.append(nn.Conv2d(16, image_channels, 3, 1, 1))
        self.conv = nn.Sequential(*self.conv)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = out.view(x.size(0), self.base_channels, self.down_size, self.down_size)
        out = self.sigmoid(self.conv(out))

        return out


class VAE(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_size, image_channels, latent_dim)
        self.decoder = Decoder(image_size, image_channels, latent_dim)
        self.device = device

    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp

        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        out = self.decoder(z)

        return out, mean, logvar

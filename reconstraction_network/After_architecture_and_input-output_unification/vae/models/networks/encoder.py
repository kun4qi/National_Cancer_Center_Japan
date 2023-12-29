import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import conv1x1
from models.blocks import ConvBlock
from models.utils import apply_init_kaiming

import torch.nn as nn
from .helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):

    def __init__(self, input_dim, z_dim, filters, activation):
        super().__init__()
        channels = filters
        attn_resolutions = []
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(input_dim, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        #layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], z_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

        self.conv1_1 = ConvBlock(z_dim, z_dim, activation=activation)
        self.conv2_1 = conv1x1(z_dim, z_dim)
        self.conv2_2 = conv1x1(z_dim, z_dim)

        apply_init_kaiming(self, activation)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, with_var=False):
        x = self.model(x)
        x = self.conv1_1(x)
        mu = self.conv2_1(x)
        logvar = self.conv2_2(x)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

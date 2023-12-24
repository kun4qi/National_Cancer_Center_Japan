import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        channels = config.model.enc_filters
        attn_resolutions = []
        num_res_blocks = 2
        resolution = 256
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(config.model.input_dim, channels[0], 3, 1, 1))

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                self.layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    self.layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                self.layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        self.layers.append(ResidualBlock(channels[-1], channels[-1]))
        #layers.append(NonLocalBlock(channels[-1]))
        self.layers.append(ResidualBlock(channels[-1], channels[-1]))
        self.layers.append(GroupNorm(channels[-1]))
        self.layers.append(Swish())
        self.layers.append(nn.Conv2d(channels[-1], config.model.latent_dim, 3, 1, 1))
        
    def forward(self, x):
        skips = []
        for layer in self.layers:
            if isinstance(layer, DownSampleBlock):
                skips.append(x)
            x = layer(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        channels = config.model.dec_filters
        attn_resolutions = []
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        self.layers = nn.ModuleList([
            nn.Conv2d(config.model.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            #NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ])

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):  
                self.layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    self.layers.append(NonLocalBlock(in_channels))
            if i != 0:
                self.layers.append(UpSampleBlock(in_channels))
                in_channels *= 2 
                resolution *= 2
        self.layers.append(GroupNorm(in_channels))
        self.layers.append(Swish())
        self.layers.append(nn.Conv2d(in_channels, config.model.output_dim, 3, 1, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x, skips):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, UpSampleBlock) and skips:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, imgs):
        encoded_images, skips = self.encoder(imgs)
        decoded_images = self.decoder(encoded_images, skips)
        return decoded_images




"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""
class Discriminator(nn.Module):
    def __init__(self, config, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(config.model.input_dim, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

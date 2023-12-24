"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn


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
        self.model = nn.Sequential(*layers)

        self.critic_network = nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size=4, stride=1, padding=4)  # output 1 channel prediction map

    def forward(self, x):
        x= self.model(x)
        #print(x.shape)

        feature = x
        feature = feature.view(feature.size(0), -1)
        x = self.critic_network(x)

        return x, feature

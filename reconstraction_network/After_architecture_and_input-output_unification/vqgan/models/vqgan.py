import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, config):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.codebook = Codebook(config)
        self.quant_conv = nn.Conv2d(config.model.latent_dim, config.model.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.model.latent_dim, config.model.latent_dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        loss = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        loss = torch.clamp(loss, 0, 1e4).detach()
        return 0.8 * loss

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))









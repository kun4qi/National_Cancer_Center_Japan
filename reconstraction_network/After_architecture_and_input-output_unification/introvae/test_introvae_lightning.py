import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from dataio import CKBrainMetDataModule
import functions.pytorch_ssim as pytorch_ssim


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        config.model.w_h,
        config.model.w_h
    )
    

class introvae(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.alpha = self.config.training.alpha
        self.beta = self.config.training.beta
        self.margin = self.config.training.margin
        self.batch_size = self.config.dataset.batch_size
        self.fixed_z = torch.randn(calc_latent_dim(self.config))
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.

        # networks
        self.E = Encoder(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.enc_filters, activation=self.config.model.enc_activation).float()
        self.D = Decoder(config).float()
        if config.model.enc_spectral_norm:
            apply_spectral_norm(self.E)
        if config.model.dec_spectral_norm:
            apply_spectral_norm(self.D)

    def l_recon(self, recon: torch.Tensor, target: torch.Tensor):
        if 'ssim' in self.config.training.loss:
            ssim_loss = pytorch_ssim.SSIM(window_size=11)

        if self.config.training.loss == 'l2':
            loss = F.mse_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'l1':
            loss = F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

        elif self.config.training.loss == 'ssim+l1':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim+l2':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.mse_loss(recon, target, reduction='sum')

        else:
            raise NotImplementedError

        return self.beta * loss / self.batch_size

    def l_reg(self, mu: torch.Tensor, log_var: torch.Tensor):
        loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
        return loss / self.batch_size
        

    def test_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.E.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.D.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        image = batch['image']
        z, _, _ = self.E(image)
        x_r = self.D(z)
        x_p = self.D(self.fixed_z.to("cuda"))

        image = image.detach().cpu()
        x_r = x_r.detach().cpu()
        x_p = x_p.detach().cpu()

        image = image[:self.config.save.n_save_images, ...]
        x_r = x_r[:self.config.save.n_save_images, ...]
        x_p = x_p[:self.config.save.n_save_images, ...]
        if self.config.save.n_save_images > self.batch_size:
            raise ValueError(f'can not save images properly')
        self.logger.test_log_images(torch.cat([image, x_r, x_p]), batch_idx, self.config.save.n_save_images)


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'E_TotalLoss', 'E_EncodeLoss', 'E_MarginLoss',
                          'E_LatentLoss', 'E_ReconLoss', 'D_TotalLoss', 'D_LatentLoss', 'D_ReconLoss',
                          'Val_E_TotalLoss', 'Val_E_EncodeLoss', 'Val_E_MarginLoss',
                          'Val_E_LatentLoss', 'Val_E_ReconLoss', 'Val_D_TotalLoss', 'Val_D_LatentLoss', 'Val_D_ReconLoss']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)

    #set callbacks
    checkpoint_callback = ModelSaver(
        limit_num=config.save.n_saved,
        save_interval=config.save.save_epoch_interval,
        monitor=None,
        dirpath=logger.log_dir,
        filename='ckpt-{epoch:04d}',
        save_top_k=-1,
        save_last=False
    )
    
    #save config
    logger.log_hyperparams(config, needs_save)

    #time per epoch
    timer = Time(config)

    dm = CKBrainMetDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        gpus=1,
        sync_batchnorm=config.training.sync_batchnorm,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="test")

    print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
    model = introvae.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
    trainer.test(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

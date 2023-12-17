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
from dataio import CKBrainMetDataModule
import functions.pytorch_ssim as pytorch_ssim

from model import VectorQuantizedVAE, to_scalar


class vqvae(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.
        self.train_lr = 0.0001

        # networks
        self.vqvae = VectorQuantizedVAE(input_dim=self.config.model.input_dim, dim=self.config.training.hidden_size, K=self.config.training.k)          # number of stepssampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])loss_type = 'l1'            # L1 or L2)

    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                #save sampling and recon image
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']
                    x_tilde, z_e_x, z_q_x = self.vqvae(image)

                    image = image.detach().cpu()
                    x_tilde = x_tilde.detach().cpu()
                    
                    image = image[:self.config.save.n_save_images, ...]
                    x_tilde = x_tilde[:self.config.save.n_save_images, ...]
                    self.logger.train_log_images(torch.cat([image, x_tilde]), self.current_epoch-1, self.config.save.n_save_images)

        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)
        
        m_optim = self.optimizers()
        m_optim.zero_grad()

        #vqvae training
        image = batch['image']

        x_tilde, z_e_x, z_q_x = self.vqvae(image)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, image)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq +  loss_commit
    
        self.manual_backward(loss)
        m_optim.step()

        if self.needs_save:
            self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('loss_recons', loss_recons, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('loss_vq', loss_vq, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('loss_commit', loss_commit, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            
        return {'loss': loss}
        

    def validation_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.vqvae.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        #save sampling and recon image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']
                    x_tilde, z_e_x, z_q_x = self.vqvae(image)

                    image = image.detach().cpu()
                    x_tilde = x_tilde.detach().cpu()
                    
                    image = image[:self.config.save.n_save_images, ...]
                    x_tilde = x_tilde[:self.config.save.n_save_images, ...]
                    self.logger.val_log_images(torch.cat([image, x_tilde]), self.current_epoch, self.config.save.n_save_images)

        #difusion training
        image = batch['image']

        x_tilde, z_e_x, z_q_x = self.vqvae(image)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, image)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq +  loss_commit

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_loss': loss.item(),
            'Val_loss_recons': loss_recons.item(),
            'Val_loss_vq': loss_vq.item(),
            'Val_loss_commit': loss_commit.item(),
            }
            self.logger.log_val_metrics(metrics)
                
        return {'Val_loss': loss}


    def configure_optimizers(self):
        m_optim = optim.Adam(filter(lambda p: p.requires_grad, self.vqvae.parameters()), self.train_lr, [0.9, 0.9999])
        return [m_optim]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'loss', 'loss_recons', 'loss_vq', 'loss_commit', 'Val_loss', 'Val_loss_recons', 'Val_loss_vq', 'Val_loss_commit']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)
    
    #save config
    logger.log_hyperparams(config, needs_save)

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

    #time per epoch
    timer = Time(config)

    dm = CKBrainMetDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        accelerator="gpu",
        devices=config.training.visible_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.training.sync_batchnorm,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = vqvae(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = vqvae.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from random import seed

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio import CKBrainMetDataModule

from torchvision import utils as vutils
from models import Discriminator
import lpips
from models import VQGAN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    
class vqgan(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.
        self.vqgan = VQGAN(config)
        self.discriminator = Discriminator(config)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = lpips.LPIPS(net='vgg')

    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                #save recon image
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0 or self.current_epoch == 0:
                    imgs = batch['image']
        
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    imgs = imgs.detach().cpu()
                    decoded_images = decoded_images.detach().cpu()

                    imgs = imgs[:self.config.save.n_save_images, ...]
                    decoded_images = decoded_images[:self.config.save.n_save_images, ...]
                    
                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                                        
                    self.logger.train_log_images(torch.cat([imgs, decoded_images]), self.current_epoch-1, self.config.save.n_save_images)
                    
        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)

        opt_vq, opt_disc = self.optimizers()

        imgs = batch['image']
    
        decoded_images, _, q_loss = self.vqgan(imgs)

        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)

        perceptual_loss = self.perceptual_loss(imgs, decoded_images).mean()
   
        if self.config.training.loss == 'l2':
            rec_loss = F.mse_loss(decoded_images, imgs, reduction='mean')

        elif self.config.training.loss == 'l1':
            rec_loss = F.l1_loss(decoded_images, imgs, reduction='mean')

        perceptual_rec_loss = self.config.training.perceptual_loss_factor * perceptual_loss + self.config.training.rec_loss_factor * rec_loss

        g_loss = -torch.mean(disc_fake) * self.config.training.g_loss

        vq_loss = perceptual_rec_loss + q_loss + g_loss

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))

        gan_loss = 0.5*(d_loss_real + d_loss_fake)

        opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)

        opt_disc.zero_grad()
        gan_loss.backward()

        opt_vq.step()
        opt_disc.step()

        torch.cuda.empty_cache()

        if self.needs_save:
            self.log('vq_loss', vq_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('gan_loss', gan_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('perceptual_loss', perceptual_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('q_loss', q_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('d_loss_real', d_loss_real, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('d_loss_fake', d_loss_fake, prog_bar=True, on_step=True, on_epoch=False, logger=True)


            
        return {'vq_loss': vq_loss, 'gan_loss' : gan_loss}
        

    def validation_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.vqgan.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        
        if self.config.training.val_mode == "train":
            for m in self.discriminator.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()


        #save recon image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    imgs = batch['image']
        
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    imgs = imgs.detach().cpu()
                    decoded_images = decoded_images.detach().cpu()

                    imgs = imgs[:self.config.save.n_save_images, ...]
                    decoded_images = decoded_images[:self.config.save.n_save_images, ...]
                   
                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                    self.logger.val_log_images(torch.cat([imgs, decoded_images]), self.current_epoch, self.config.save.n_save_images)


        #difusion training
        imgs = batch['image']
        
        decoded_images, _, q_loss = self.vqgan(imgs)

        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)

        perceptual_loss = self.perceptual_loss(imgs, decoded_images).mean()
        
        if self.config.training.loss == 'l2':
            rec_loss = F.mse_loss(decoded_images, imgs, reduction='mean')

        elif self.config.training.loss == 'l1':
            rec_loss = F.l1_loss(decoded_images, imgs, reduction='mean')

        perceptual_rec_loss = self.config.training.perceptual_loss_factor * perceptual_loss + self.config.training.rec_loss_factor * rec_loss

        g_loss = -torch.mean(disc_fake)

        #lamda = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        #vq_loss = perceptual_rec_loss + q_loss + lamda * g_loss
        vq_loss = perceptual_rec_loss + q_loss + g_loss


        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = 0.5*(d_loss_real + d_loss_fake)

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_vq_loss': vq_loss.item(),
            'Val_gan_loss': gan_loss.item(),
            'Val_perceptual_loss': perceptual_loss.item(),
            'Val_rec_loss': rec_loss.item(),
            'Val_q_loss': q_loss.item(),
            'Val_g_loss': g_loss.item(),
            'Val_d_loss_real': d_loss_real.item(),
            'Val_d_loss_fake': d_loss_fake.item(),
            }
            self.logger.log_val_metrics(metrics)

                
        return {'vq_loss': vq_loss, 'gan_loss' : gan_loss}


    def configure_optimizers(self):

        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=self.config.model.vq_lr, eps=1e-08, betas=(0.9, 0.999)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.config.model.disc_lr, eps=1e-08, betas=(0.9, 0.999))
        return [opt_vq, opt_disc]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'vq_loss', 'gan_loss', 'perceptual_loss', 'rec_loss', 'q_loss', 'g_loss', 'd_loss_real', 'd_loss_fake', 
                          'Val_vq_loss', 'Val_gan_loss', 'Val_perceptual_loss', 'Val_rec_loss', 'Val_q_loss', 'Val_g_loss', 'Val_d_loss_real', 'Val_d_loss_fake']
  
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
        deterministic=True,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = vqgan(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = vqgan.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

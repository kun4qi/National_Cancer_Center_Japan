import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy

from models import Discriminator
from models import Generator
from utils import load_json
from utils import check_manual_seed
from utils import norm
from utils import denorm
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio.settings import TRAIN_PATIENT_IDS
from dataio.settings import TEST_PATIENT_IDS
from dataio import MNISTDataModule
from torch.optim.lr_scheduler import CosineAnnealingLR


class Anogan(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.

        # networks
        self.D = Discriminator(config)
        self.G = Generator(config)


    def anomaly_score(self, input_image, fake_image, D):
        # Residual loss の計算
        residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

        total_loss_by_image = residual_loss
        total_loss = total_loss_by_image.sum()

        return total_loss, total_loss_by_image, residual_loss

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss


    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
        )
        return d_loss

    def criterion(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        return loss(y_hat, y)


    def test_step(self, batch, batch_idx):
        #if self.config.training.val_mode == "train":
        #    for m in self.G.modules():
        #        if isinstance(m, nn.BatchNorm2d):
        #            m.train()

        #    for m in self.D.modules():
        #        if isinstance(m, nn.BatchNorm2d):
        #            m.train()

        image = batch['image']
        
        z = torch.randn(image.size(0), self.config.model.latent_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
        z.requires_grad = True
        z_optimizer = torch.optim.Adam([z], lr=self.config.training.z_lr)
        scheduler = CosineAnnealingLR(z_optimizer, T_max=50)
        with torch.enable_grad():
            residual_loss_list = []

            for epoch in range(self.config.training.z_epochs):
                #z探し
                fake_images = self.G(z)
                loss, _, residual_loss = self.anomaly_score(image, fake_images, self.D)
                z_optimizer.zero_grad()
                loss.backward()
                z_optimizer.step()
                scheduler.step()
                residual_loss_list.append(residual_loss[0].cpu().detach().numpy())
        
        pd.DataFrame({'residual_loss':residual_loss_list}).to_csv(f'{self.config.save.output_root_dir}{self.config.save.study_name}/version_0/z_loss_{batch_idx}.csv', index=False)
        
        #異常度の計算
        fake_images = self.G(z)

        z_0 = torch.randn(image.size(0), self.config.model.latent_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
        sample_images = self.G(z_0)

        image = image.detach().cpu()
        fake_images = fake_images.detach().cpu()
        sample_images = sample_images.detach().cpu()

        image = image[:self.config.save.n_save_images, ...]
        fake_images = fake_images[:self.config.save.n_save_images, ...]
        sample_images = sample_images[:self.config.save.n_save_images, ...]
        if self.config.save.n_save_images > self.batch_size:
            raise ValueError(f'can not save images properly')
        self.logger.test_log_images(torch.cat([image, fake_images, sample_images]), batch_idx, self.config.save.n_save_images)


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'G_Loss', 'D_Loss', 'Val_G_Loss', 'Val_D_Loss']
  
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

    dm = MNISTDataModule(config)

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
    model = Anogan.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
    trainer.test(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

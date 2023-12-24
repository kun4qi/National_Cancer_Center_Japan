import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from utils import weights_init_normal
from model import Generator
from model import Discriminator
from dataio import CKBrainMetDataModule

    

class cyclegan(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.batch_size = self.config.dataset.batch_size
        # Networks A is normal, B is abnormal
        self.netG_A2B = Generator(self.config)
        self.netG_B2A = Generator(self.config)
        self.netD_A = Discriminator(self.config)
        self.netD_B = Discriminator(self.config)
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        # Lossess
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.


    def test_step(self, batch, batch_idx):
        if self.config.training.val_mode == "train":
            for m in self.netG_A2B.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.netG_B2A.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.netD_A.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.netD_B.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        
        size = batch['normal_image'].size(0)

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.config.training.use_cuda else torch.Tensor
        input_A = Tensor(size, self.config.dataset.input_ch, self.config.dataset.image_size, self.config.dataset.image_size)
        input_B = Tensor(size, self.config.dataset.output_ch, self.config.dataset.image_size, self.config.dataset.image_size)
        target_real = Variable(Tensor(size).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(size).fill_(0.0), requires_grad=False)

        # Set model input
        real_A = Variable(input_A.copy_(batch['normal_image']))
        real_B = Variable(input_B.copy_(batch['abnormal_image']))


        ###### Generators A2B and B2A ######

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = self.netG_A2B(real_B)
        loss_identity_B = self.criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = self.netG_B2A(real_A)
        loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = self.netG_A2B(real_A)
        pred_fake = self.netD_B(fake_B)
        loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

        fake_A = self.netG_B2A(real_B)
        pred_fake = self.netD_A(fake_A)
        loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = self.netG_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = self.netG_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        ###################################

        ###### Discriminator A ######
        # Real loss
        pred_real = self.netD_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = self.netD_A(fake_A.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        ###################################

        ###### Discriminator B ######
        # Real loss
        pred_real = self.netD_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, target_real)
        
        # Fake loss
        pred_fake = self.netD_B(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        ###################################
                
        real_A = real_A.detach().cpu()
        fake_B = fake_B.detach().cpu()
        real_B = real_B.detach().cpu()
        fake_A = fake_A.detach().cpu()

        real_A = real_A[:self.config.save.n_save_images, ...]
        fake_B = fake_B[:self.config.save.n_save_images, ...]
        real_B = real_B[:self.config.save.n_save_images, ...]
        fake_A = fake_A[:self.config.save.n_save_images, ...]
        if self.config.save.n_save_images > self.batch_size:
            raise ValueError(f'can not save images properly')
        self.logger.test_log_images(torch.cat([real_A, fake_B, real_B, fake_A]), batch_idx, self.config.save.n_save_images)


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'loss_G', 'loss_G_identity', 'loss_G_GAN',
                          'loss_G_cycle', 'loss_D', 'Val_loss_G', 'Val_loss_G_identity', 'Val_loss_G_GAN',
                          'Val_loss_G_cycle', 'Val_loss_D']
  
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
    model = cyclegan.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
    trainer.test(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

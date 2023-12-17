import argparse
import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio import CKBrainMetDataModule

from models import Network


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    
class palette(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.
        self.netG = Network(config)
        if self.config.training.loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif self.config.training.loss_type == 'l1':
            self.loss = nn.L1Loss
        else:
            raise ValueError(f'{self.config.training.loss_type} is invalid')
        self.netG.set_loss(self.loss)
        self.netG.set_new_noise_schedule()
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                #save recon image
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0 or self.current_epoch == 0:
                    imgs, masks = batch['image'], batch['mask']

                    _, recon_image = self.netG(imgs, y_cond=masks)

                    imgs = imgs.detach().cpu()
                    masks = masks.detach().cpu()
                    recon_image = recon_image.detach().cpu()

                    imgs = imgs[:self.config.save.n_save_images, ...]
                    masks = masks[:self.config.save.n_save_images, ...]
                    recon_image = recon_image[:self.config.save.n_save_images, ...]

                    masks = masks.repeat(1, 3, 1, 1)  # now masks has shape [batch_size, 3, height, width]
                    
                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                                        
                    self.logger.train_log_images(torch.cat([masks, recon_image, imgs]), self.current_epoch-1, self.config.save.n_save_images)
                    
        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)

        opt = self.optimizers()

        imgs, masks = batch['image'], batch['mask']

        loss, _ = self.netG(imgs, y_cond=masks)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if self.needs_save:
            self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            
        return {'loss': loss}
        

    def validation_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.netG.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        #save recon image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    imgs, masks = batch['image'], batch['mask']

                    _, recon_image = self.netG(imgs, y_cond=masks)

                    imgs = imgs.detach().cpu()
                    masks = masks.detach().cpu()
                    recon_image = recon_image.detach().cpu()

                    imgs = imgs[:self.config.save.n_save_images, ...]
                    masks = masks[:self.config.save.n_save_images, ...]
                    recon_image = recon_image[:self.config.save.n_save_images, ...]

                    masks = masks.repeat(1, 3, 1, 1)  # now masks has shape [batch_size, 3, height, width]

                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                    self.logger.val_log_images(torch.cat([masks, recon_image, imgs]), self.current_epoch, self.config.save.n_save_images)

        imgs, masks = batch['image'], batch['mask']

        loss, _ = self.netG(imgs, y_cond=masks)

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_loss': loss.item(),
            }
            self.logger.log_val_metrics(metrics)
                
        return {'loss': loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), lr=self.config.training.lr, eps=1e-08, betas=(0.9, 0.999))
        return [opt]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'loss', 'Val_loss']
  
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
      model = palette(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = palette.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)

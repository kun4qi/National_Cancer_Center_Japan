import numpy as np
import os
import re
from glob import glob
from PIL import Image
import cv2
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule

from .transformation import Normalize
from .transformation import ToImage
from .transformation import ToTensor
from .transformation import RandomHorizontalFlip
from .transformation import Resize


class CKBrainMetDataset(Dataset):

    def __init__(self, config, mode, data_root_paths, transform, image_size):
        super().__init__()
        assert mode in ['train', 'test']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        """
        self.config = config
        self.mode = mode
        self.data_root_paths = data_root_paths
        self.transform = transform
        self.image_size = image_size
        self.files = self.build_file_paths()

    def build_file_paths(self):

        if self.mode == 'train':
            image_paths = glob(os.path.join(self.data_root_paths + "/image/train2017/*jpg"))
            mask_paths = glob(os.path.join(self.data_root_paths + "/mask/train2017/*png"))
            image_paths = natsorted(image_paths)
            mask_paths = natsorted(mask_paths)
            return list(zip(image_paths, mask_paths))
        
        elif self.mode == 'test':
            image_paths = glob(os.path.join(self.data_root_paths + "/image/val2017/*jpg"))
            mask_paths = glob(os.path.join(self.data_root_paths + "/mask/val2017/*png"))
            image_paths = natsorted(image_paths)
            mask_paths = natsorted(mask_paths)
            return list(zip(image_paths, mask_paths))
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = cv2.imread(self.files[index][0])
        mask = cv2.imread(self.files[index][1], cv2.IMREAD_UNCHANGED)

        sample = {
        'image': image,
        'mask': mask,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class CKBrainMetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir_path = self.config.dataset.root_dir_path
        self.CKBrainMetDataset = CKBrainMetDataset
        self.omit_transform = False

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            if self.config.dataset.use_augmentation:
                transform = transforms.Compose([
                    ToImage(),
                    Resize(self.config),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(),
                ])
            else:
                transform = transforms.Compose([
                    ToImage(),
                    Resize(self.config),
                    ToTensor(),
                    Normalize(),
                ])

            val_transform = transforms.Compose([
                    ToImage(),
                    Resize(self.config),
                    ToTensor(),
                    Normalize(),
                ])

            if self.omit_transform:
                transform = None
            
            self.train_dataset = self.CKBrainMetDataset(config=self.config, mode='train', data_root_paths=self.root_dir_path, transform=transform, image_size=self.config.dataset.image_size)
            self.valid_dataset = self.CKBrainMetDataset(config=self.config, mode='train', data_root_paths=self.root_dir_path, transform=val_transform, image_size=self.config.dataset.image_size)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            transform = transforms.Compose([
                    ToImage(),
                    Resize(self.config),
                    ToTensor(),
                    Normalize(),
                ])

            self.test_dataset = self.CKBrainMetDataset(config=self.config, mode='test', data_root_paths=self.root_dir_path, transform=transform, image_size=self.config.dataset.image_size)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.dataset.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)
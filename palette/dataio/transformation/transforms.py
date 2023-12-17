import torch
import random
import numpy as np
from PIL import Image


class Normalize(object):
    """Normalizes image with range of 0-255 to 0-1.
    """

    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample: dict):
        image = sample['image']
        image -= self.min_val
        image /= (self.max_val - self.min_val)
        image = torch.clamp(image, 0, 1)

        mask = sample['mask']
        mask -= self.min_val
        mask /= (self.max_val - self.min_val)
        mask = torch.clamp(mask, 0, 1)

        sample.update({
            'image': image,
            'mask': mask,
        })

        return sample
    
class Resize(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample: dict):
        image = sample['image']
        image = image.resize((self.config.dataset.image_size, self.config.dataset.image_size))

        mask = sample['mask']
        mask = mask.resize((self.config.dataset.image_size, self.config.dataset.image_size))


        sample.update({
            'image': image,
            'mask': mask,
        })

        return sample


class ToImage(object):

    def __call__(self, sample):
        # assert 'label' not in sample.keys()
        image = sample['image']
        mask = sample['mask']

        sample.update({
            'image': Image.fromarray(image),
            'mask': Image.fromarray(mask),
        })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict):
        image = sample['image']

        if type(image) == Image.Image:
            image = np.asarray(image)

        if image.ndim == 2:
            image = image[np.newaxis, ...]
        
        image = np.transpose(image, (2,0,1))

        image = torch.from_numpy(image).float()
        sample.update({
            'image': image,
        })

        mask = sample['mask']

        if type(mask) == Image.Image:
            mask = np.asarray(mask)

        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        mask = torch.from_numpy(mask).float()
        sample.update({
            'mask': mask,
        })

        if 'label' in sample.keys():
            label = sample['label']

            if label.ndim == 2:
                label = label[np.newaxis, ...]

            label = torch.from_numpy(label).int()
            sample.update({
                'label': label,
            })

        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        mask = sample['mask']

        if random.random() < 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample.update({
            'image': image,
            'mask': mask,
        })

        return sample
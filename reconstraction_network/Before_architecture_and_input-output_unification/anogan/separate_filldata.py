from .transformation import ToImage
import argparse
import numpy as np
import os
import re
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import numpy as np
from PIL import Image
    

class ToTensor(object):

    def __call__(self, sample: dict):
        image = sample['image']

        if type(image) == Image.Image:
            image = np.asarray(image)

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        #image = torch.from_numpy(image).float()
        sample.update({
            'image': image,
        })

        if 'label' in sample.keys():
            label = sample['label']

            if label.ndim == 2:
                label = label[np.newaxis, ...]

            #label = torch.from_numpy(label).int()
            sample.update({
                'label': label,
            })

        return sample


class CKBrainMetDataset(Dataset):

    def __init__(self, mode, patient_paths, transform):
        super().__init__()
        assert mode in ['train', 'test']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        """
        self.mode = mode
        self.patient_paths = patient_paths
        self.transform = transform
        self.files = self.build_file_paths(self.patient_paths)

    def build_file_paths(self, patient_paths):

        files = []

        for patient_path in patient_paths:
            file_paths = glob(os.path.join(patient_path + "/*" + 'flair' + ".npy")) #指定のスライスのパスを取得
            for file_path in file_paths:
                
                if 'Abnormal' in file_path:
                    class_name = 'Abnormal'
                else:
                    #assert 'normal' in file_name
                    class_name = 'Normal'

                patient_id = patient_path.split('/')[-1]
                file_name = file_path.split('/')[-1]
                study_name = self.get_study_name(patient_path)
                slice_num = self.get_slice_num(file_name)
                path_name = patient_path
                

                if self.mode == 'train':
                    files.append({
                        'image': file_path,
                        'path_name': path_name,
                        'file_name': file_name,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

                elif self.mode == 'test' or self.mode == 'test_normal':
                    label_path = self.get_label_path(file_path)

                    files.append({
                        'image': file_path,
                        'path_name': path_name,
                        'file_name': file_name,
                        'label': label_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

        return files

    def get_study_name(self, patient_path):
        study_name = patient_path.split('/')[-3]
        return study_name
    
    def get_slice_num(self, file_name):
        n = re.findall(r'\d+', file_name) #image_fileのスライス番号の取り出し
        return n[-1]

    def get_label_path(self, file_path):
        file_path = file_path.replace(self.config.dataset.select_slice, 'seg')
        return file_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = np.load(self.files[index]['image'])
        image = np.flipud(np.transpose(image))

        sample = {
            'image': image.astype(np.float32),
            'path_name': self.files[index]['path_name'],
            'file_name': self.files[index]['file_name'],
            'patient_id': self.files[index]['patient_id'],
            'class_name': self.files[index]['class_name'],
            'study_name': self.files[index]['study_name'],
            'slice_num': self.files[index]['slice_num'],
        }

        return sample

def get_patient_paths(base_dir_path):
        patient_ids = os.listdir(base_dir_path)
        return [os.path.join(base_dir_path, p) for p in patient_ids]


def main(th):

    transform = transforms.Compose([
                        ToImage(),
                        ToTensor(),
                    ])

    root_dir_path = './data/brats_separated'

    train_patient_paths = get_patient_paths(os.path.join(root_dir_path, 'MICCAI_BraTS_2019_Data_Val_Testing/Normal'))
    train_dataset = CKBrainMetDataset(mode='train', patient_paths=train_patient_paths, transform=transform)

    for i in tqdm(range(len(train_dataset))):
        if np.count_nonzero(train_dataset[i]['image'] <= 0) <=th:
            os.makedirs(train_dataset[i]['path_name'].replace("brats_separated", f"fill_data_{th}"), exist_ok=True)
            np.save(train_dataset[i]['path_name'].replace("brats_separated", f"fill_data_{th}") + '/' + train_dataset[i]['file_name'], train_dataset[i]['image'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-th', '--th', help='Which thresholds to filter by', required=True)
    args = parser.parse_args()

    main(args.th)



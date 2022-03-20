import numpy as np 
import os, sys
import matplotlib.pyplot as plt 

import numpy as np
import os 
import json  
import pandas as pd 
from PIL import Image

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfms

path_to_unlabeled_data = 'stl10/stl10_binary/unlabeled_X.bin'

def normalize_transform():
    return tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class STLUnlabeledDataloader(Dataset):
    def __init__(self, dataroot):

        self.dataroot = dataroot 
        with open(path_to_unlabeled_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
        self.images = np.reshape(everything, (-1, 3, 96, 96))
        

        self.byol_tfms = tfms.Compose([
                            tfms.RandomHorizontalFlip(),
                            tfms.ToPILImage(),
                            tfms.RandomApply([
                                    tfms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                            ], p=0.8),
                            tfms.RandomGrayscale(p=0.2),
                            tfms.ToTensor(),
                            normalize_transform()
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = torch.from_numpy(self.images[idx])

        # byol data augmentation
        online_transformed = self.byol_tfms(img_tensor)
        target_transformed = self.byol_tfms(img_tensor)

        return {'online_input': online_transformed, 'target_input': target_transformed}


class STLLabeledDataloader(Dataset):
    def __init__(self, dataroot, mode='train'):

        self.dataroot = dataroot 
        self.mode = mode 

        DATA_PATH = os.path.join(dataroot, '{}_X.bin'.format(mode))
        LABEL_PATH = os.path.join(dataroot, '{}_y.bin'.format(mode))

        with open(LABEL_PATH, 'rb') as f:
            self.labels = np.fromfile(f, dtype=np.int_)

        with open(DATA_PATH, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8)
        self.images = np.reshape(images, (-1, 3, 96, 96))

        # print("images:", self.images.shape)
        # print("labels:", self.labels.shape)

        self.classification_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.RandomHorizontalFlip(),
            normalize_transform()
        ])

        self.val_tfms = tfms.Compose([
            tfms.ToTensor(),
            normalize_transform()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = torch.from_numpy(self.images[idx])
        label = self.labels[idx]

        if self.mode == 'train':
            transformed = self.classification_tfms(img_tensor)
        else:
            transformed = self.val_tfms(img_tensor)

        return {'input': transformed, 'label': label}

# from torchvision.datasets import STL10


# classification_tfms = tfms.Compose([
#             tfms.ToTensor(),
#             tfms.RandomHorizontalFlip(),
#             normalize_transform()
        # ])

# train_ds = STL10(root='stl10/', split='train', transform=classification_tfms)
# print('train')
# print('classes:', train_ds.classes)
# print('data shape', train_ds.data.shape)

# trainDataloader = DataLoader(train_ds, batch_size=16, shuffle=True)

# for batch in trainDataloader:
#     print(batch[0].shape)
#     print(batch[1])
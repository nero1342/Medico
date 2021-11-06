import torch
from torch.utils.data import Dataset, DataLoader
import cv2

import os
from os import path

from PIL import Image
import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    """
    """
    def __init__(self, root, subset = 'train'):
        self.root = root
        self.img_root = path.join(root, 'images') 
        self.mask_root = path.join(root, 'masks')

        self.subset = subset 
        
        img_list = sorted(os.listdir(self.img_root))
        self.images = img_list # []

        print('%d out of %d images accepted in %s.' % (len(self.images), len(img_list), self.img_root))

        # Final transform without randomness
        if subset == 'train':
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Affine(rotate = [-90, 90]), 
                    A.RandomResizedCrop(480, 480, scale=(0.8, 1.5)),
                    A.Resize(384, 384),

                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(384, 384),
                    ToTensorV2(),
                ]
            )

    def __getitem__(self, idx):
        image = self.images[idx]
        info = {}
        info['name'] = image

        im_path = path.join(self.img_root, image)
        msk_path = path.join(self.mask_root, image)
        if not path.isfile(msk_path):
          msk_path = msk_path.replace('.jpg', '.png')
        # this_im = cv2.imread(im_path)
        # this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)
        this_im = (np.array(Image.open(im_path).convert('RGB')) / 255).astype(np.float32)
        
        try:        
            this_mask = np.array(Image.open(msk_path).convert('P')) // 200
        except Exception as e:
            # print(e)
            this_mask = np.zeros(this_im.shape[:2]) 
        
        info['size'] = this_mask.shape
        if self.transform:
            sample = self.transform(image=this_im, mask=this_mask)
            this_im, this_mask = sample['image'], sample['mask']

        this_mask = this_mask.long()
        out = {
            'img': this_im,
            'mask': this_mask,
            'info': info
        }
        return out

    def __len__(self):
        return len(self.images)

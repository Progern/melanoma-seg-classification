import os
import pandas as pd
import torch
import albumentations as alb
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import constants as const

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform

    
class MelanomaSegmentationDataset(Dataset):
    """
    Provides a convenient interface to create
    a dataset of images and corresponding masks
    """

    def __init__(self, csv_file, root_dir, augmentation=None, preprocessing=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_csv = csv_file
        self.image_dir, self.masks_dir = root_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing 

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.data_csv.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = os.path.join(self.masks_dir, self.data_csv.iloc[idx, 0])
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.expand_dims(mask, -1)
        
        # Calculate unique values and check if mask 
        # is in range of 0-1
        mask_unique = np.unique(mask)
        if mask_unique[mask_unique > 1].any():
            mask = mask // 255
        

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask'] 
            

        return image, mask
    
    
    @staticmethod
    def get_default_transformation():
        return alb.Compose([
                                alb.HorizontalFlip(p=0.5),
                                alb.VerticalFlip(p=0.5),
                                alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                                                     rotate_limit=30, p=0.4, border_mode=0)
                            ])
    
    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    
    @staticmethod
    def get_default_preprocessing(encoder: str = const.ENCODER,
                                  encoder_weights: str = const.ENCODER_WEIGHTS):
        """Construct preprocessing transform
    
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """
        
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 
                                                             encoder_weights)
        
        
        _transform = [
            alb.Lambda(image = preprocessing_fn),
            alb.Lambda(image = MelanomaSegmentationDataset.to_tensor, 
                       mask = MelanomaSegmentationDataset.to_tensor),
        ]
        
        return alb.Compose(_transform)

    
    
        
import os
import torch
import pandas as pd
import numpy as np
import constants as const
import segmentation_models_pytorch as smp
import datetime

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_loader import MelanomaSegmentationDataset


def get_train_val_dataset_segmentation(train_data_path, images_root, masks_root, random_seed: int = const.random_seed_const):
    """
    Loads the training data, splits the dataset into train/val
    and returns two dataset instances for training and validation
    """
    
    train_data = pd.read_csv(train_data_path)
    
    train_data_split, val_data_split = train_test_split(train_data, 
                                                       random_state = random_seed, 
                                                       train_size = const.train_val_split_size,
                                                       shuffle = True)
    
    train_dataset = MelanomaSegmentationDataset(csv_file = train_data_split, 
                                                root_dir = (images_root, masks_root),
                                                augmentation = MelanomaSegmentationDataset.get_default_transformation(),
                                                preprocessing = MelanomaSegmentationDataset.get_default_preprocessing())


    validation_dataset = MelanomaSegmentationDataset(csv_file = val_data_split, 
                                                     root_dir = (images_root, masks_root),
                                                     preprocessing = MelanomaSegmentationDataset.get_default_preprocessing())
    
    return train_dataset, validation_dataset

def get_test_dataset_segmentation(test_data_path, images_root, masks_root, random_seed: int = const.random_seed_const):
    """
    Loads the testing data and returns one dataset 
    instance for testing
    """
    
    test_data = pd.read_csv(test_data_path)
    test_dataset = MelanomaSegmentationDataset(csv_file = test_data, 
                                               root_dir = (images_root, masks_root),
                                               preprocessing = MelanomaSegmentationDataset.get_default_preprocessing())
    
    return test_dataset


def get_data_loader(dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 1):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

def generate_model_name(architecture: str = "unet", encoder: str = "resnet_34"):
    return "{}_backbone_{}_{}".format(architecture, encoder, datetime.datetime.now())

def load_data(data_root):
    # Paths to image-mask pairs
    images_root = os.path.join(data_root, "images_all")
    masks_root = os.path.join(data_root, "masks_all")
    
    # Load metadata for images and masks
    train_data_dist = os.path.join(data_root, "train_data.csv")
    
    return get_train_val_dataset_segmentation(train_data_dist, 
                                              images_root, 
                                              masks_root)

def get_train_epoch(model, loss, metrics, optimizer, device: str = const.DEVICE):
    return smp.utils.train.TrainEpoch(model, 
                                      loss=loss, 
                                      metrics=metrics, 
                                      optimizer=optimizer,
                                      device=device,
                                      verbose=True)

def get_valid_epoch(model, loss, metrics, device: str = const.DEVICE):
    return smp.utils.train.ValidEpoch(model, 
                                     loss=loss, 
                                     metrics=metrics, 
                                     device=const.DEVICE,
                                     verbose=True)

def create_model(encoder: str = const.ENCODER):
    return smp.FPN(encoder_name = encoder, 
                   encoder_weights = const.ENCODER_WEIGHTS, 
                   classes = len(const.CLASSES), 
                   activation = const.ACTIVATION)

def get_loss():
    return smp.utils.losses.DiceLoss()

def get_metrics():
    return [smp.utils.metrics.IoU(threshold=0.5)]

def get_optimizator(model, 
                    opt_type: str, 
                    lr: float, 
                    momentum: float, 
                    weight_decay: float):
    
    if opt_type == "sgd":
        return torch.optim.SGD([dict(params=model.parameters(), 
                                     lr = lr, 
                                     momentum = momentum, 
                                     weight_decay = weight_decay)])
    elif opt_type == 'adam':
        return torch.optim.AdamW([dict(params=model.parameters(), 
                                      lr = lr, 
                                      momentum = momentum, 
                                      weight_decay = weight_decay)])
    else:
        raise ValueError("Unrecognized optimizer type {}".format(opt_type))


# Borrowed from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# With some minor fixing from my side
class EarlyStopping(object):
    def __init__(self, mode='max', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
       
        metrics = torch.from_numpy(np.array(metrics))
        
        try:
            if self.best is None:
                self.best = metrics
                return False

            if torch.isnan(metrics):
                return True

            if self.is_better(metrics, self.best):
                self.num_bad_epochs = 0
                self.best = metrics
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                return True
        except Exception as ex:
            print("Error occured in the Early Stopping Callback.")
            print(ex)

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
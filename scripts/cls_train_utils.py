import sys
import os
import datetime
import torch
import pandas as pd
import time
import copy
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pytorch_lightning.metrics.classification import F1

from data_loader import MelanomaClassificationDataset
from seg_train_utils import get_data_loader, EarlyStopping

import constants as const



def load_data(data_root):
    images_root = os.path.join(data_root, "images_all_processed")

    train_data_dist = os.path.join(data_root, "train_data.csv")
    train_data = pd.read_csv(train_data_dist)
    
    train_data = train_data.replace({"class": {"benign": 0, "malignant": 1}})
    
    print("We have {} benign data points".format(len(train_data[train_data["class"] == 0])))
    print("We have {} malignant data points".format(len(train_data[train_data["class"] == 1])))
    
    train_data_split, val_data_split = train_test_split(train_data, 
                                                        random_state = const.random_split_const, 
                                                        train_size = const.train_val_split_size,
                                                        shuffle = True)
    
    train_dataset = MelanomaClassificationDataset(csv_file = train_data_split, 
                                                  root_dir = images_root,
                                                  augmentation = MelanomaClassificationDataset\
                                                      .get_default_transformation(),
                                                  preprocessing = MelanomaClassificationDataset\
                                                      .get_default_preprocessing())


    validation_dataset = MelanomaClassificationDataset(csv_file = val_data_split, 
                                                       root_dir = images_root,
                                                       preprocessing = MelanomaClassificationDataset\
                                                           .get_default_preprocessing())
    
    return train_dataset, train_data_split, validation_dataset


def get_class_weights(train_data_split, device, 
                      underclass_idx: int = 1, 
                      underclass_scale_factor: int = 1):
    num_samples_benign = len(train_data_split[train_data_split["class"] == 0]) 
    num_samples_malignant = len(train_data_split[train_data_split["class"] == 1]) 
    num_samples_general = [num_samples_benign, num_samples_malignant]

    normed_weights = [1 - (x / sum(num_samples_general)) for x in num_samples_general]
    
    # Additionally give bigger weight to the undersampled class
    normed_weights[underclass_idx] = normed_weights[underclass_idx] * underclass_scale_factor
    
    normed_weights = torch.FloatTensor(normed_weights).to(device)
    
    return normed_weights

def get_default_class_weights(device):
    return torch.FloatTensor([1, 1]).to(device)


def get_data_loaders(train_dataset, validation_dataset):
    train_loader = get_data_loader(train_dataset, 
                                   batch_size = const.batch_size_train, 
                                   shuffle=True, num_workers = 0)
    
    val_loader = get_data_loader(validation_dataset, 
                                 batch_size = const.batch_size_val, 
                                 shuffle=False, num_workers = 0)
    
    return train_loader, val_loader


def generate_model_name(architecture: str = "resnet-34"):
    return "{}_{}".format(architecture, datetime.datetime.now())

def train_model(model, dataloaders, dataset_sizes,
                criterion, optimizer, metric, scheduler, 
                architecture, device, num_epochs = 25):
    
    since = time.time()
    model_name = '../models/{}.pth'.format(generate_model_name(architecture))
    print(model_name)

    best_f1 = 0.0
    es = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1 = 0.0

            print("{} phase".format(phase))
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    f1 = metric(outputs.cpu(), labels.cpu())
                    running_f1 += f1

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                epoch_f1 = running_f1 / (dataset_sizes[phase]  / 8)
            else:
                epoch_f1 = running_f1 / dataset_sizes[phase]

            print('{} Loss: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_f1))

            # Save best model weights
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model, model_name)
                
            # Early stopping
            if es.step(best_f1):
                break  

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best F1: {:4f}'.format(best_f1))


def get_model(architecture, normed_weights, device):
    num_classes = len(const.CL_CLASSES)
    
    if architecture == "inception":
        model_ft = models.inception_v3(pretrained=True)
        model_ft.aux_logits = False
    elif architecture == "resnet":
        model_ft = models.resnet18(pretrained=True)
    elif architecture == "googlenet":
        model_ft = models.googlenet(pretrained=True)
    else:
        raise ValueError("Unknown architecture!")
        
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(const.CL_CLASSES))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(normed_weights)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler



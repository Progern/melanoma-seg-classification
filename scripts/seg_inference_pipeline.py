import os
import sys
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

import constants as const

from tqdm import tqdm
from argparse import ArgumentParser

from train_utils import *
from metrics import get_iou_metric

def main(model_path):
    data_root = "../dataset"
    
    print("Loading data...")
    data = get_data(data_root)
    
    print("Restoring model state...")
    model = get_model(model_path)
    
    print("Establishing data pipeline...")
    loader = get_data_loader_int(data, 
                                 batch_size = const.batch_size_val, 
                                 shuffle = False, 
                                 num_workers = 0)
    
    print("Predictions...")
    preds, gts, ious = predict(model, loader._get_iterator(), data.__len__())
    
    ious_arr = np.array(ious)
    ious_arr = ious_arr[ious_arr > 0]
    print("Average IoU: {}".format(np.mean(ious_arr)))
    
    
def get_data(data_root):
    # Define paths
    images_root = os.path.join(data_root, "images_all")
    masks_root = os.path.join(data_root, "masks_all")
    
    # Load metadata file
    test_data_dist = os.path.join(data_root, "test_data.csv")
    
    return get_test_dataset_segmentation(test_data_dist, images_root, masks_root)
    
def get_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def get_data_loader_int(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return get_data_loader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

def iou_numpy(outputs: np.array, labels: np.array):
    intersection = np.logical_and(labels, outputs)
    union = np.logical_or(labels, outputs)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def predict(model, iterator, num_images):
    preds = []
    gts = []
    ious = []
    idx = 0

    try:
        with torch.no_grad(): 
            current_sample = iterator.next()
            
            while current_sample is not None:
                if idx <= 1300:
                    idx += 1
                    image, gt = current_sample
                    image, gt = image.cuda(), gt.cuda()

                    # Perform prediction            
                    pred = model(image)

                    # Transform outputs
                    pred_cpu = pred.cpu().numpy()
                    gt_cpu = gt.cpu().numpy()

                    # Post-process the results
                    pred_cpu[pred_cpu >= 0.5] = 1
                    pred_cpu[pred_cpu < 0.5] = 0

                    pred_cpu = pred_cpu.astype(int)
                    gt_cpu = gt_cpu.astype(int)

                    iou = iou_numpy(pred_cpu, gt_cpu)
                    print("Processing image {} / {}. IoU: {}".format(idx, num_images, iou))
                    ious.append(iou)

                    # Save results
                    gts.append(gt_cpu)
                    preds.append(pred_cpu)

                    current_sample = iterator.next()
                else:
                    break
        
        return preds, gts, ious
    except StopIteration as ex:
        pass
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str,
                    help="path to the model weights and metadata")
    
    parser.add_argument("-dp", "--data_path", type=str,
                    help="path to the dataset")
    
    args = parser.parse_args()
    path = args.path if args.path is not None else ""
    data_path = args.data_path if args.data_path is not None else ""
    
    path = os.path.join("../models", path)
    main(path, data_path)
    
"""
Modified from:
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

from functools import partial
from typing import Callable

import torch
import numpy as np


def get_iou_metric(num_classes: int,
                  ignore: int = None,
                  per_image: bool = False) -> Callable[[torch.Tensor, torch.Tensor], float]:
    if num_classes == 2:
        return partial(_iou_binary, per_image=per_image)
    else:
        return partial(_iou, C=num_classes, ignore=ignore, per_image=per_image)

    
def _mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _iou_binary(preds, labels, EMPTY=1., per_image=False, thresh=0.0):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    preds = ((preds.float().view(-1, 1, 1, 1)).data > thresh).float()
    labels = labels.view(-1, 1, 1, 1)
    
    if not per_image:
        preds, labels = (preds,), (labels,)
    
    ious = []
    
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            iou = EMPTY
        else:
            iou = intersection.item() / union.item()
        ious.append(iou)
    iou = _mean(ious)
    return 100 * iou


def _iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    preds = ((torch.softmax(preds, axis=1).argmax(axis=1).float().view(-1, 1, 1, 1)).data).float()
    labels = labels.view(-1, 1, 1, 1)
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i)))
                if ignore is not None:
                    union = union & (label != ignore)
                union = union.sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [_mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * _mean(np.array(ious))

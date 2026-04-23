import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def binary_confusion_counts(output, target, threshold=0.5):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    pred = output > threshold
    truth = target > threshold

    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, np.logical_not(truth)).sum()
    fn = np.logical_and(np.logical_not(pred), truth).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum()

    return tp, fp, fn, tn


def binary_segmentation_metrics(tp, fp, fn, tn, smooth=1e-5):
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    bg_iou = (tn + smooth) / (tn + fp + fn + smooth)
    miou = (iou + bg_iou) / 2

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'miou': miou,
    }

import torch
from typing import Sequence, Optional, Union
from torch.autograd.variable import Variable
import torch.nn as nn


def make_one_hot(labels, C=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    if len(labels.shape) == 3:
        labels = labels.unsqueeze(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def diceCoeffv2(pred, gt, eps=1e-5):
    r"""
    Requires activated pred and gt
    """
    n_classes = pred.shape[1]
    with torch.no_grad():
        gt = make_one_hot(gt.long(), C=n_classes)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return diceCoeffv2(y_pr, y_gt)

# Inspired by
# https://github.com/BBillot/SynthSeg/blob/492453421020d66ebf0e11bf0cc266754d21b895/SynthSeg/evaluate.py
import numpy as np
import torch


def faster_dice(x, y, labels, fudge_factor=1e-8):
    """Faster PyTorch implementation of Dice scores.
    :param x: input label map as torch.Tensor
    :param y: input label map as torch.Tensor of the same size as x
    :param labels: list of labels to evaluate on
    :param fudge_factor: an epsilon value to avoid division by zero
    :return: pytorch Tensor with Dice scores in the same order as labels.
    """

    assert (
        x.shape == y.shape
    ), "both inputs should have same size, had {} and {}".format(
        x.shape, y.shape
    )

    if len(labels) > 1:
        dice_score = torch.zeros(len(labels))
        for label in labels:
            x_label = x == label
            y_label = y == label
            xy_label = (x_label & y_label).sum()
            dice_score[label] = (
                2 * xy_label / (x_label.sum() + y_label.sum() + fudge_factor)
            )

    else:
        dice_score = dice(
            x == labels[0], y == labels[0], fudge_factor=fudge_factor
        )

    return dice_score


def dice(x, y, fudge_factor=1e-8):
    """Implementation of dice scores ofr 0/1 numy array"""
    return 2 * torch.sum(x * y) / (torch.sum(x) + torch.sum(y) + fudge_factor)


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.nn.functional.softmax(inputs, dim=1)

        dice_loss = 0.0
        for c in range(inputs.size(1)):
            true_flat = (targets == c).float()
            pred_flat = inputs[:, c]
            intersection = (pred_flat * true_flat).sum(dim=(1, 2, 3))
            total = (pred_flat + true_flat).sum(dim=(1, 2, 3))

            # Adding the smooth term in the denominator
            dice_loss += 1 - (2.0 * intersection + smooth) / (total + smooth)

        return dice_loss.mean()


class DiceLossInt(torch.nn.Module):
    def __init__(self):
        super(DiceLossInt, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Getting the number of classes from inputs
        num_classes = torch.max(inputs) + 1

        dice_loss = 0.0
        for c in range(num_classes.long()):
            true_flat = (targets == c).float()
            pred_flat = (inputs == c).float()
            intersection = (pred_flat * true_flat).sum(dim=(1, 2, 3))
            total = (pred_flat + true_flat).sum(dim=(1, 2, 3))

            # Adding the smooth term in the denominator
            dice_loss += 1 - (2.0 * intersection + smooth) / (total + smooth)

        return dice_loss.mean()

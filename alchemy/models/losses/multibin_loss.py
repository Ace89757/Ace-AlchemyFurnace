# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import torch

from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from alchemy.registry import MODELS


def multibin_loss(pred_orientations: Tensor, gt_orientations: Tensor, num_dir_bins: int) -> Tensor:
    """
    Multi-Bin Loss.

    Args:
        pred_orientations(Tensor): Predicted local vector orientation in [axis_cls, head_cls, sin, cos] format. shape (N, num_dir_bins * 4)
        gt_orientations(Tensor): Corresponding gt bboxes, shape (N, num_dir_bins * 2).
        num_dir_bins(int): Number of bins to encode direction angle. Defaults to 4.

    Returns:
        Tensor: Loss tensor.
    """
    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    
    for i in range(num_dir_bins):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(pred_orientations[:, (i * 2):(i * 2 + 2)], gt_orientations[:, i].long(), reduction='mean')

        # regression loss
        valid_mask_i = (gt_orientations[:, i] == 1)
        cls_losses += cls_ce_loss

        if valid_mask_i.sum() > 0:
            start = num_dir_bins * 2 + i * 2
            end = start + 2
            pred_offset = F.normalize(pred_orientations[valid_mask_i, start:end])
            gt_offset_sin = torch.sin(gt_orientations[valid_mask_i, num_dir_bins + i])
            gt_offset_cos = torch.cos(gt_orientations[valid_mask_i, num_dir_bins + i])
            reg_loss = F.mse_loss(pred_offset[:, 0], gt_offset_sin, reduction='none') + F.mse_loss(pred_offset[:, 1], gt_offset_cos, reduction='none')
            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

    return cls_losses / num_dir_bins + reg_losses / (reg_cnt + 1e-6)


@MODELS.register_module()
class AlchemyMultiBinLoss(nn.Module):
    """
    Multi-Bin Loss for orientation.

    Args:
        reduction (str): The method to reduce the loss. Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction: str = 'none', loss_weight: float = 1.0) -> None:
        super(AlchemyMultiBinLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor, num_dir_bins: int) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            num_dir_bins (int): Number of bins to encode direction angle.
            reduction_override (str, optional): The reduction method used to override the original reduction method of the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """

        loss = self.loss_weight * multibin_loss(pred, target, num_dir_bins)
        return loss

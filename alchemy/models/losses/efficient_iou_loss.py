# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from mmdet.models.losses.utils import weighted_loss

from alchemy.registry import MODELS


@weighted_loss
def efficient_iou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area 最小外接矩形框
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]**2
    ch = enclose_wh[:, 1]**2

    c2 = cw + ch + eps   # 最小外接矩形框对角线

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    pred_xs = (pred[:, 0] + pred[:, 2]) / 2
    pred_ys = (pred[:, 1] + pred[:, 3]) / 2

    target_xs = (target[:, 0] + target[:, 2]) / 2
    target_ys = (target[:, 1] + target[:, 3]) / 2

    # 尺寸损失
    w_dist = (w1 - w2)**2 / (cw + eps)
    h_dist = (h1 - h2)**2 / (ch + eps)

    # 中心点距离损失
    d = (pred_xs - target_xs) ** 2 + (pred_ys - target_ys) ** 2  # 求中心点的欧式距离，即论文中的d
    c_dist = d / c2

    # eious
    eious = 1 - ious + c_dist + w_dist + h_dist

    return eious


@MODELS.register_module()
class EfficientIoULoss(nn.Module):
    """
    `Implementation of paper Focal and Efficient IOU Loss for Accurate Bounding Box Regression <https://arxiv.org/pdf/2101.08158.pdf>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean', loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """
        Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2), shape (n, 4).
            target (Tensor): The learning target of the prediction, shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method used to override the original reduction method of the loss.
                                                          Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0

        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future reduce the weight of shape (n, 4) to (n,) to match the giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
            
        loss = self.loss_weight * efficient_iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
            )

        return loss

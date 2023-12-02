# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.

from .multibin_loss import AlchemyMultiBinLoss
from .efficient_iou_loss import EfficientIoULoss


__all__ = [
    'AlchemyMultiBinLoss', 
    'EfficientIoULoss'
]
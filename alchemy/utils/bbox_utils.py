# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

from torch import Tensor
from numpy import ndarray
from typing import Optional, Union


def cal_bboxes_area(bboxes: Union[Tensor, ndarray], keep_axis: Optional[bool] = False) -> Union[Tensor, ndarray]:
    """
    计算bbox的面积
    
    Args:
        bboxes (Tensor or ndarray): [n, (x1, y1, x2, y2, .....)]
    """
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    
    if keep_axis:
        return areas[:, None]

    return areas
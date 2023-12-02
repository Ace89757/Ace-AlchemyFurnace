# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import torch
import numpy as np

from typing import List
from torch import Tensor

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

from alchemy.registry import TASK_UTILS


@TASK_UTILS.register_module()
class CenterNetMono3dCoder(BaseBBoxCoder):
    def __init__(self, num_dir_bins: int, bin_centers: List[float], bin_margin: float = 0.1, use_box_type: bool = False, **kwargs):
        super().__init__(use_box_type, **kwargs)
        assert num_dir_bins >= 1
        self.num_dir_bins = num_dir_bins
        self.bin_margin = bin_margin
        self.bin_centers = [float(x * np.pi) for x in bin_centers]

    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes) -> Tensor:
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (`BaseInstance3DBoxes`): Ground truth 3D bboxes. shape: (N, 7).

        Returns:
            torch.Tensor: Targets of orientations.
        """
        local_yaw = gt_bboxes_3d.local_yaw  # [-pi, pi]

        # encode local yaw (-pi ~ pi) to multibin format
        encode_local_yaw = local_yaw.new_zeros([local_yaw.shape[0], self.num_dir_bins * 2])

        bin_size = 2 * np.pi / self.num_dir_bins
        margin_size = bin_size * self.bin_margin

        bin_centers = local_yaw.new_tensor(self.bin_centers)
        range_size = bin_size / 2 + margin_size
        
        # 计算alpha和每个center的offset
        offsets = local_yaw.unsqueeze(1) - bin_centers.unsqueeze(0)

        # 判断alpha落在哪个bin内
        for i in range(self.num_dir_bins):
            offset = offsets[:, i]
            inds = abs(offset) < range_size
            encode_local_yaw[inds, i] = 1
            encode_local_yaw[inds, i + self.num_dir_bins] = offset[inds]

        orientation_target = encode_local_yaw

        return orientation_target

    def decode(self, dir_bin_pred: Tensor) -> Tensor:
        """
        dir_bin_pred: [n, self.num_dir_bins * 4]
        """
        pred_bin_cls = dir_bin_pred[:, :self.num_dir_bins * 2].view(-1, self.num_dir_bins, 2)
        pred_bin_cls = pred_bin_cls.softmax(dim=2)[..., 1]
        orientations = dir_bin_pred.new_zeros(dir_bin_pred.shape[0])

        # 计算属于哪个bin
        bin_cls = pred_bin_cls.argmax(dim=1)

        for i in range(self.num_dir_bins):
            mask_i = (bin_cls == i)   # 找到输入该bin的目标
            start_bin = self.num_dir_bins * 2 + i * 2

            end_bin = start_bin + 2
            pred_bin_offset = dir_bin_pred[mask_i, start_bin:end_bin]
            
            orientations[mask_i] = pred_bin_offset[:, 0].atan2(pred_bin_offset[:, 1]) + self.bin_centers[i]

        orientations = orientations.reshape(-1, 1)

        return orientations

    def decode_location(self, pred_ctxs, pred_ctys, pred_depths, cam2img):
        """
        pred_ctxs: [n, 1]
        pred_ctys: [n, 1]
        pred_depths: [n, 1]
        """

        # 1. 计算2d框的中心点坐标
        pred_ctxs = pred_ctxs.reshape((-1, 1))   # [n, 1]
        pred_ctys = pred_ctys.reshape((-1, 1))   # [n, 1]

        # 2. 计算相机坐标系下的坐标
        x = (pred_ctxs - cam2img[0][2]) * pred_depths / cam2img[0][0]
        y = (pred_ctys - cam2img[1][2]) * pred_depths / cam2img[1][1]

        locations = torch.cat([x, y, pred_depths], dim=1)

        return locations

    def decode_orientation(self, pred_local_yaws, pred_locations):
        """
        pred_local_yaws: [n, 1]
        """
        pred_locations = pred_locations.view(-1, 3)
        rays = pred_locations[:, 0].atan2(pred_locations[:, 2])
        yaws = pred_local_yaws + rays.reshape(-1, 1)

        # 转到[-pi, pi]之间
        larger_idx = (yaws > np.pi).nonzero(as_tuple=False)
        small_idx = (yaws < -np.pi).nonzero(as_tuple=False)
        if len(larger_idx) != 0:
            yaws[larger_idx] -= 2 * np.pi
            
        if len(small_idx) != 0:
            yaws[small_idx] += 2 * np.pi
        
        return yaws

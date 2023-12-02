# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Optional, Tuple

from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import (ConfigType, InstanceList, OptConfigType, OptInstanceList, OptMultiConfig)

from alchemy.registry import MODELS
from alchemy.utils import cal_bboxes_area


@MODELS.register_module()
class AlchemyTTFNet(BaseDenseHead):
    """
    Training-Time-Friendly Network for Real-Time Object Detection.
    Paper link <https://arxiv.org/abs/1909.00700>
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 beta: float = 0.54,
                 alpha: float = 0.54,
                 down_ratio: int = 4,
                 bbox_convs: int = 2,
                 base_anchor: int = 16,
                 heatmap_convs: int = 2,
                 bbox_channels: int = 64,
                 heatmap_channels: int = 128,
                 bbox_gaussian: bool = True,
                 bbox_area_process: str = 'log',
                 loss_bbox: ConfigType = dict(type='EfficientIoULoss', loss_weight=5.0),
                 loss_center_heatmap: ConfigType = dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 test_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.beta = beta
        self.alpha = alpha
        self.base_loc = None
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.bbox_convs = bbox_convs
        self.down_ratio = down_ratio
        self.base_anchor = base_anchor
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.heatmap_convs =heatmap_convs
        self.bbox_gaussian = bbox_gaussian
        self.bbox_channels = bbox_channels
        self.heatmap_channels = heatmap_channels
        self.bbox_area_process = bbox_area_process

        # post process
        self.topk = 100 if self.test_cfg is None else self.test_cfg.get('topk', 100)
        self.score_thr = 0.5 if self.test_cfg is None else self.test_cfg.get('score_thr', 0.5)        
        self.local_maximum_kernel = 3 if self.test_cfg is None else self.test_cfg.get('local_maximum_kernel', 3)

        # initial heads
        self._initial_heads()
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)

        self.fp16_enabled = False

    def _initial_heads(self):
        """
        Initialize heads.
        """
        self.heatmap_head = self._build_head(self.num_classes, self.heatmap_convs, self.heatmap_channels)
        self.bbox_head = self._build_head(4, self.bbox_convs, self.bbox_channels)

    def _build_head(self, out_channels, conv_num, feat_channels):
        """
        Build head.

        Args:
            out_channels (int): The channels of output feature map.
            conv_num (int): Number of convs.
            feat_channels (int): The channels of conv feature map.
        
        Return:
            head: (Sequential[module]): The module of head.
        """
        head_convs = []
        
        for i in range(conv_num):
            inp = self.in_channels if i == 0 else feat_channels
            head_convs.append(ConvModule(inp, feat_channels, 3, padding=1))

        head_convs.append(nn.Conv2d(feat_channels, out_channels, 1))

        return nn.Sequential(*head_convs)

    def init_weights(self) -> None:
        """
        Initialize weights of the head.
        """
        for _, m in self.heatmap_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.heatmap_head[-1], std=0.01, bias=bias_cls)

        for _, m in self.bbox_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        """
        Forward features. Notice CenterNet head does not use FPN.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            bbox_pred (Tensor): wh predicts, the channels number is 4.
        """
        feat = x[0]
        heatmap_pred = self.heatmap_head(feat).sigmoid()
        bbox_pred = F.relu(self.bbox_head(feat)) * self.base_anchor

        return (heatmap_pred, bbox_pred)

    def loss_by_feat(self, 
                     heatmap_pred: Tensor, 
                     bbox_pred: Tensor, 
                     batch_gt_instances: InstanceList, 
                     batch_img_metas: List[dict], 
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        Compute losses of the head.

        Args:
            heatmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): bbox predicts for all levels with shape (B, 4, H, W).
            batch_gt_instances (list[:obj:'InstanceData']): Batch of gt_instance. It usually includes 'bboxes' and 'labels' attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:'InstanceData'], optional): Batch of gt_instances_ignore. 
                                                                             It includes 'bboxes' attribute data that is ignored during training and testing. 
                                                                             Defaults to None.

        Returns:
            dict[str, Tensor]: which has components below
                - loss_heatmap (Tensor): loss of center heatmap.
                - loss_bbox (Tensor): loss of left right top bottom
        """
        feat_h, feat_w = heatmap_pred.shape[2:]   # [n, c, h, w]
        target_result = self._get_targets(batch_gt_instances, (feat_h, feat_w))

        # heatmap loss
        center_heatmap_targets = target_result['heatmap_targets']
        center_heatmap_avg_factor = max(center_heatmap_targets.eq(1).sum(), 1)
        heatmap_pred = torch.clamp(heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_center_heatmap(heatmap_pred, center_heatmap_targets, avg_factor=center_heatmap_avg_factor)

        # bbox loss
        if (self.base_loc is None) or (feat_h != self.base_loc.shape[1]) or (feat_w != self.base_loc.shape[2]):
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (feat_w - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_pred.device)
            shifts_y = torch.arange(0, (feat_h - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        pred_boxes = torch.cat((self.base_loc - bbox_pred[:, [0, 1]], self.base_loc + bbox_pred[:, [2, 3]]), dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = target_result['bbox_targets'].permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target_weights = target_result['bbox_target_weights'].permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_avg_factor = torch.sum(bbox_target_weights[:, 0]) + 1e-6

        loss_bbox = self.loss_bbox(pred_boxes, bbox_targets, bbox_target_weights, avg_factor=bbox_avg_factor)

        return dict(loss_heatmap=loss_center_heatmap, loss_bbox=loss_bbox)

    @staticmethod
    def _calc_region(bbox, ratio):
        """
        Calculate a proportional bbox region.

        The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

        Args:
            bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
            ratio (float): Ratio of the output region.

        Returns:
            tuple: x1, y1, x2, y2
        """
        x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
        y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
        x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
        y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()

        return (x1, y1, x2, y2)

    def _gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        """
        Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma_x (int): Sigma X of gaussian function. Default: 1.
            sigma_y (int): Sigma Y of gaussian function. Default: 1.

        Returns:
            h (Tensor): Gaussian kernel with a '(2 * radius + 1) * (2 * radius + 1)' shape.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        """
        Generate 2D gaussian heatmap.

        Args:
            heatmap (Tensor): Input heatmap, the gaussian kernel will cover on it and maintain the max value.
            center (list[int]): Coord of gaussian kernel's center.
            h_radius (int): Radius h of gaussian kernel.
            w_radius (int): Radius w of gaussian kernel.
            k (int): Coefficient of gaussian kernel. Default: 1.

        Returns:
            out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
        """
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self._gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap
    
    def _get_targets_single(self, gt_instances: InstanceData, feat_shape: tuple):
        """
        Compute regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - bbox_target (Tensor): targets of bbox predict, shape (4, H, W).
               - bbox_target_weight (Tensor): weights of bbox predict, shape (4, H, W).
        """
        feat_h, feat_w = feat_shape
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        # init targets
        fake_heatmap = gt_bboxes.new_zeros((feat_h, feat_w))
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        bbox_target = gt_bboxes.new_ones((4, feat_h, feat_w)) * -1
        bbox_target_weight = gt_bboxes.new_zeros((4, feat_h, feat_w))

        # 计算bbox的面积
        bbox_areas = cal_bboxes_area(gt_bboxes)

        # 减小大、小目标的影响
        if self.bbox_area_process == 'log':
            bbox_areas = bbox_areas.log()
        elif self.bbox_area_process == 'sqrt':
            bbox_areas = bbox_areas.sqrt()

        # 按面积大小排序(从大到小)
        num_objs = bbox_areas.size(0)
        bbox_areas_sorted, boxes_ind = torch.topk(bbox_areas, num_objs)

        if self.bbox_area_process == 'norm':
            bbox_areas_sorted[:] = 1.

        gt_bboxes = gt_bboxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        # down sample
        feat_bboxes = gt_bboxes.clone() / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        # 目标的中心点(在输出图上)
        feat_ctx_ints = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2 / self.down_ratio).to(torch.int)
        feat_cty_ints = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2 / self.down_ratio).to(torch.int)

        # 目标的高斯半径
        h_radiuses_alpha = (feat_bboxes_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_bboxes_ws / 2. * self.alpha).int()
        if self.bbox_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_bboxes_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_bboxes_ws / 2. * self.beta).int()

        # 如果bbox的标签不使用高斯范围的，计算每个目标的中心区域，此区域内的点作为正例点
        if not self.bbox_gaussian:
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = self._calc_region(gt_bboxes.transpose(0, 1), r1)

            ctr_x1s = torch.round(ctr_x1s.float() / self.down_ratio).int()
            ctr_y1s = torch.round(ctr_y1s.float() / self.down_ratio).int()
            ctr_x2s = torch.round(ctr_x2s.float() / self.down_ratio).int()
            ctr_y2s = torch.round(ctr_y2s.float() / self.down_ratio).int()

            ctr_x1s, ctr_x2s = [torch.clamp(x, max=feat_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=feat_h - 1) for y in [ctr_y1s, ctr_y2s]]

        for idx in range(num_objs):
            cat_id = gt_labels[idx]
            
            # heatmap
            fake_heatmap = fake_heatmap.zero_()
            self._draw_truncate_gaussian(
                fake_heatmap, 
                (feat_ctx_ints[idx], feat_cty_ints[idx]), 
                h_radiuses_alpha[idx].item(), 
                w_radiuses_alpha[idx].item()
                )
            
            # 若两个目标的高斯范围有重叠区域，选取大的值作为标签(对小目标友好)
            heatmap_target[cat_id] = torch.max(heatmap_target[cat_id], fake_heatmap)

            # bbox的权重
            # TODO: 好像是回归权重没有考虑优先小目标？
            if self.bbox_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self._draw_truncate_gaussian(
                        fake_heatmap, 
                        (feat_ctx_ints[idx], feat_cty_ints[idx]),
                        h_radiuses_beta[idx].item(),
                        w_radiuses_beta[idx].item()
                        )
                bbox_target_inds = fake_heatmap > 0

                local_heatmap = fake_heatmap[bbox_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= bbox_areas_sorted[idx]
                bbox_target_weight[:, bbox_target_inds] = local_heatmap / ct_div

            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[idx], ctr_y1s[idx], ctr_x2s[idx], ctr_y2s[idx]
                bbox_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                bbox_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                bbox_target_weight[:, bbox_target_inds] = bbox_areas_sorted[idx] / bbox_target_inds.sum().float()
            
            # 回归的是输入图上的尺寸
            bbox_target[:, bbox_target_inds] = gt_bboxes[idx][:, None]
            
        return heatmap_target, bbox_target, bbox_target_weight

    def _get_targets(self, batch_gt_instances: InstanceList, feat_shape: tuple) -> Tuple[dict, int]:
        """
        Compute regression and classification targets in multiple images.

        Args:
            batch_gt_instances (InstanceList): batch of gt_instance. It usually includes 'bboxes' and 'labels' attributes.
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - bbox_target (Tensor): targets of wh predict, shape (4, H, W).
               - bbox_target_weight (Tensor): targets of offset predict, shape (1, H, W).
        """
        feat_h, feat_w = feat_shape

        with torch.no_grad():
            heatmap_target, bbox_target, bbox_target_weight = multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                feat_shape=(feat_h, feat_w)
            )
        
        heatmap_targets, bbox_targets, bbox_target_weights = [
            torch.stack(t, dim=0).detach() for t in [
                heatmap_target, 
                bbox_target, 
                bbox_target_weight
                ]]

        target_result = dict(
            heatmap_targets=heatmap_targets,
            bbox_targets=bbox_targets,
            bbox_target_weights=bbox_target_weights
            )

        return target_result

    def predict_by_feat(self, heatmap_pred: Tensor, bbox_pred: Tensor, batch_img_metas: Optional[List[dict]] = None, rescale: bool = True) -> InstanceList:
        """
        Transform network output for a batch into bbox predictions.

        Args:
            heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): WH predicts with shape (B, 4, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info. Defaults to None.
            rescale (bool): If True, return boxes in original image space. Defaults to True.
            with_nms (bool): If True, do nms before return boxes. Defaults to False.

        Returns:
            list[:obj:'InstanceData']: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        topk_bboxes, topk_scores, topk_labels = self._decode(heatmap_pred, bbox_pred)  # [n, topk, 4], [n, topk], [n, topk]

        # 解析每个batch的结果
        result_list = []
        for idx in range(topk_bboxes.shape[0]):
            scores = topk_scores[idx]
            keep = scores > self.score_thr

            if sum(keep):
                img_meta = batch_img_metas[idx]

                det_bboxes = topk_bboxes[idx][keep]
                
                # rescale -> raw
                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

                det_scores = scores[keep]
                det_labels = topk_labels[idx][keep]
            else:
                det_bboxes = torch.zeros((0, 4), dtype=torch.float32)
                det_scores = torch.zeros((0, ), dtype=torch.float32)
                det_labels = torch.zeros((0, ), dtype=torch.float32)

            result = InstanceData()
            result.bboxes = det_bboxes
            result.scores = det_scores
            result.labels = det_labels

            result_list.append(result)

        return result_list

    def _decode(self, heatmap_pred: Tensor, bbox_pred: Tensor):
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): BBox predicts with shape (B, 4, H, W).

        Returns:
            Tuple[Tensor]: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - topk_bboxes (Tensor): Has a shape (B, topk, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - topk_scores (Tensor): Classification scores, has a shape (B, topk)
                - topk_labels (Tensor): Labels of bboxes, has a shape (B, topk).      
        """
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == heatmap_pred).float()
        heatmap_pred = heatmap_pred * keep

        # topk
        batch_size, num_classes, output_h, output_w = heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点
        topk_xs = topk_xs.view(-1, self.topk, 1) * self.down_ratio
        topk_ys = topk_ys.view(-1, self.topk, 1) * self.down_ratio

        # bbox
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()   # [n, h, w, c]
        bbox_pred = bbox_pred.view(batch_size, -1, 4)  # [n, h*w, c]
        topk_indexes = topk_indexes.unsqueeze(2).expand(batch_size, topk_indexes.size(1), 4)
        bbox_pred = bbox_pred.gather(1, topk_indexes).view(-1, self.topk, 4)

        topk_bboxes = torch.cat([
            topk_xs - bbox_pred[..., [0]], 
            topk_ys - bbox_pred[..., [1]],
            topk_xs + bbox_pred[..., [2]],
            topk_ys + bbox_pred[..., [3]]
            ], dim=2)
    
        return topk_bboxes, topk_scores, topk_labels
    

@MODELS.register_module()
class AlchemyTTFNetPlus(AlchemyTTFNet):
    def _get_targets_single(self, gt_instances: InstanceData, feat_shape: tuple):
        """
        Compute regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - bbox_target (Tensor): targets of bbox predict, shape (4, H, W).
               - bbox_target_weight (Tensor): weights of bbox predict, shape (4, H, W).
        """
        feat_h, feat_w = feat_shape
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        # init targets
        fake_heatmap = gt_bboxes.new_zeros((feat_h, feat_w))
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        bbox_target = gt_bboxes.new_ones((4, feat_h, feat_w)) * -1
        bbox_target_weight = gt_bboxes.new_zeros((4, feat_h, feat_w))

        # 计算bbox的面积
        bbox_areas = cal_bboxes_area(gt_bboxes)

        # 减小大、小目标的影响
        if self.bbox_area_process == 'log':
            bbox_areas = bbox_areas.log()
        elif self.bbox_area_process == 'sqrt':
            bbox_areas = bbox_areas.sqrt()

        # 按面积大小排序(从大到小)
        num_objs = bbox_areas.size(0)
        bbox_areas_sorted, boxes_ind = torch.topk(bbox_areas, num_objs)

        if self.bbox_area_process == 'norm':
            bbox_areas_sorted[:] = 1.

        gt_bboxes = gt_bboxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        # down sample
        feat_bboxes = gt_bboxes.clone() / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        # 目标的中心点(在输出图上)
        feat_ctx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2 / self.down_ratio
        feat_ctx_ints = feat_ctx.to(torch.int)
        feat_ctx_offset = feat_ctx - feat_ctx_ints
        feat_ctx_offset[feat_ctx_offset < 0.5] = -1
        feat_ctx_offset[feat_ctx_offset >= 0.5] = 1

        feat_cty = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2 / self.down_ratio
        feat_cty_ints = feat_cty.to(torch.int)
        feat_cty_offset = feat_cty - feat_cty_ints
        feat_cty_offset[feat_cty_offset < 0.5] = -1
        feat_cty_offset[feat_cty_offset >= 0.5] = 1

        # 目标的高斯半径
        h_radiuses_alpha = (feat_bboxes_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_bboxes_ws / 2. * self.alpha).int()
        if self.bbox_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_bboxes_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_bboxes_ws / 2. * self.beta).int()

        # 如果bbox的标签不使用高斯范围的，计算每个目标的中心区域，此区域内的点作为正例点
        if not self.bbox_gaussian:
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = self._calc_region(gt_bboxes.transpose(0, 1), r1)

            ctr_x1s = torch.round(ctr_x1s.float() / self.down_ratio).int()
            ctr_y1s = torch.round(ctr_y1s.float() / self.down_ratio).int()
            ctr_x2s = torch.round(ctr_x2s.float() / self.down_ratio).int()
            ctr_y2s = torch.round(ctr_y2s.float() / self.down_ratio).int()

            ctr_x1s, ctr_x2s = [torch.clamp(x, max=feat_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=feat_h - 1) for y in [ctr_y1s, ctr_y2s]]

        for idx in range(num_objs):
            obj_h, obj_w = feat_bboxes_hs[idx].item(), feat_bboxes_ws[idx].item()

            # 保证在输入图上至少有1x1个像素表示
            if obj_h < 0.25 or obj_w < 0.25:
                continue

            cat_id = gt_labels[idx]
            ctx, cty = feat_ctx_ints[idx], feat_cty_ints[idx]
            
            # heatmap
            fake_heatmap = fake_heatmap.zero_()
            self._draw_truncate_gaussian(fake_heatmap, (ctx, cty), h_radiuses_alpha[idx].item(), w_radiuses_alpha[idx].item())

            # 若两个目标的高斯范围有重叠区域，选取大的值作为标签(对小目标友好)
            heatmap_target[cat_id] = torch.max(heatmap_target[cat_id], fake_heatmap)

            # bbox权重
            if self.bbox_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self._draw_truncate_gaussian(fake_heatmap, (ctx, cty), h_radiuses_beta[idx].item(), w_radiuses_beta[idx].item())

                bbox_target_inds = fake_heatmap > 0

                local_heatmap = fake_heatmap[bbox_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= bbox_areas_sorted[idx]
                bbox_target_weight[:, bbox_target_inds] = torch.max(bbox_target_weight[:, bbox_target_inds], local_heatmap / ct_div)
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[idx], ctr_y1s[idx], ctr_x2s[idx], ctr_y2s[idx]
                bbox_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                bbox_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                bbox_target_weight[:, bbox_target_inds] = bbox_areas_sorted[idx] / bbox_target_inds.sum().float()
                
            # 回归的是输入图上的尺寸
            bbox_target[:, bbox_target_inds] = gt_bboxes[idx][:, None]

            # 添加额外的正样本
            if obj_h > 1 and obj_w > 1:
                # 左右
                extra_ctx = min(max(ctx + feat_ctx_offset[idx], 0), feat_w - 1)
                heatmap_target[cat_id, int(cty), int(extra_ctx)] = 1

                # 上下
                extra_cty = min(max(cty + feat_cty_offset[idx], 0), feat_h - 1)
                heatmap_target[cat_id, int(extra_cty), int(ctx)] = 1

        return heatmap_target, bbox_target, bbox_target_weight
# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Optional, Tuple

from mmcv.cnn import ConvModule
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init
from mmdet3d.models.dense_heads.base_mono3d_dense_head import BaseMono3DDenseHead
from mmdet.utils import (ConfigType, InstanceList, OptConfigType, OptInstanceList, OptMultiConfig)
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target, multi_apply, transpose_and_gather_feat)

from alchemy.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class AlchemyCenterNetMono3d(BaseMono3DDenseHead):
    """
    Objects as Points Head.
    Paper link <https://arxiv.org/abs/1904.07850>
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 down_ratio: int = 4,
                 stacked_convs: int = 2,
                 feat_channels: int = 128,
                 class_agnostic: bool = True,
                 orientation_bin_margin: float = 0.3,
                 orientation_centers: List[float] = [0.5, -0.5],
                 loss_wh: ConfigType = dict(type='mmdet.L1Loss', loss_weight=0.1),
                 loss_depth: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_size3d: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_offset: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_orientation: ConfigType = dict(type='AlchemyMultiBinLoss', loss_weight=1.0),
                 loss_center_heatmap: ConfigType = dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 test_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.class_agnostic = class_agnostic
        self.num_dir_bins = len(orientation_centers)
        self.wh_dim = 2 if self.class_agnostic else 2 * num_classes

        # post process
        self.topk = 100 if self.test_cfg is None else self.test_cfg.get('topk', 100)
        self.score_thr = 0.5 if self.test_cfg is None else self.test_cfg.get('score_thr', 0.5)        
        self.local_maximum_kernel = 3 if self.test_cfg is None else self.test_cfg.get('local_maximum_kernel', 3)

        # initial heads
        self._initial_heads()
        self.loss_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_wh = MODELS.build(loss_wh)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_depth = MODELS.build(loss_depth)
        self.loss_size3d = MODELS.build(loss_size3d)
        self.loss_dir = MODELS.build(loss_orientation)

        # coder
        self.bbox_coder = TASK_UTILS.build(dict(type='CenterNetMono3dCoder', num_dir_bins=self.num_dir_bins, bin_centers=orientation_centers, bin_margin=orientation_bin_margin))

        self.fp16_enabled = False
    
    def _initial_heads(self):
        """
        Initialize heads.
        """
        self.heatmap_head = self._build_head(self.num_classes)
        self.wh_head = self._build_head(self.wh_dim)
        self.offset_head = self._build_head(2)
        self.depth_head = self._build_head(1)
        self.size3d_head = self._build_head(3)
        self.dir_head = self._build_head(self.num_dir_bins * 4)

    def _build_head(self, out_channels):
        """
        Build head.

        Args:
            out_channels (int): The channels of output feature map.
        
        Return:
            head: (Sequential[module]): The module of head.
        """
        head_convs = []
        
        for i in range(self.stacked_convs):
            inp = self.in_channels if i == 0 else self.feat_channels
            head_convs.append(ConvModule(inp, self.feat_channels, 3, padding=1))

        head_convs.append(nn.Conv2d(self.feat_channels, out_channels, 1))

        return nn.Sequential(*head_convs)
    
    def init_weights(self) -> None:
        """
        Initialize weights of the head.
        """
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.wh_head, self.offset_head, self.depth_head, self.size3d_head, self.dir_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
    
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        """
        Forward features.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            bbox_pred (Tensor): wh predicts, the channels number is wh_dim.
            offset_pred (Tensor): offset predicts, the channels number is 2.
            depth_pred (Tensor): depth predicts, the channels number is 1.
            size3d_pred (Tensor): size3d predicts, the channels number is 3.
            dir_pred (Tensor): orientation predicts, the channels number is num_bins * 4.
        """
        feat = x[0]
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        depth_pred = self.depth_head(feat).sigmoid()
        size3d_pred = self.size3d_head(feat)
        dir_pred = self.dir_head(feat)

        return (center_heatmap_pred, wh_pred, offset_pred, depth_pred, size3d_pred, dir_pred)
    
    def loss_by_feat(self, 
                     center_heatmap_pred: Tensor, 
                     wh_pred: Tensor, 
                     offset_pred: Tensor,
                     depth_pred: Tensor,
                     size3d_pred: Tensor,
                     dir_pred: Tensor,
                     batch_gt_instances_3d: InstanceList,
                     batch_gt_instances: InstanceList, 
                     batch_img_metas: List[dict], 
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        Compute losses of the head.

        Args:
            center_heatmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (list[Tensor]): wh predicts for all levels with shape (B, wh_dim, H, W).
            offset_pred (list[Tensor]): offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            batch_gt_instances (list[:obj:'InstanceData']): Batch of gt_instance. It usually includes ''bboxes'' and ''labels'' attributes.

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
                - loss_depth (Tensor): loss of depth heatmap.
                - loss_size3d (Tensor): loss of size3d heatmap.
                - loss_dir (Tensor): loss of orientation heatmap.
        """
        target_result = self._get_targets(batch_gt_instances, batch_gt_instances_3d, center_heatmap_pred.shape[2:])

        # heatmap loss
        center_heatmap_targets = target_result['heatmap_targets']
        center_heatmap_avg_factor = max(center_heatmap_targets.eq(1).sum(), 1)
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_heatmap(center_heatmap_pred, center_heatmap_targets, avg_factor=center_heatmap_avg_factor)

        # wh loss
        wh_targets = target_result['wh_targets']
        wh_target_weights = target_result['wh_target_weights']
        wh_avg_factors = max(wh_target_weights.eq(1).sum(), 1)
        loss_wh = self.loss_wh(wh_pred, wh_targets, wh_target_weights, avg_factor=wh_avg_factors)

        # offset loss
        offset_targets = target_result['offset_targets']
        offset_target_weights = target_result['offset_target_weights']
        offset_avg_factors = max(offset_target_weights.eq(1).sum(), 1)
        loss_offset = self.loss_offset(offset_pred, offset_targets, offset_target_weights, avg_factor=offset_avg_factors)

        # depth loss
        depth_pred = 1 / depth_pred - 1.
        depth_targets = target_result['depth_targets']
        depth_target_weights = target_result['depth_target_weights']
        depth_avg_factors = max(depth_target_weights.eq(1).sum(), 1)
        loss_depth = self.loss_depth(depth_pred, depth_targets, depth_target_weights, avg_factor=depth_avg_factors)

        # size3d loss
        size3d_targets = target_result['size3d_targets']
        size3d_target_weights = target_result['size3d_target_weights']
        size3d_avg_factors = max(size3d_target_weights.eq(1).sum(), 1)
        loss_size3d = self.loss_size3d(size3d_pred, size3d_targets, size3d_target_weights, avg_factor=size3d_avg_factors)

        # oirentation loss
        dir_pred = dir_pred.permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 4)
        dir_bin_targets = target_result['dir_bin_targets'].permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 2)
        loss_dir = self.loss_dir(dir_pred, dir_bin_targets, self.num_dir_bins)

        return dict(
            loss_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size3d=loss_size3d,
            loss_dir=loss_dir
            )

    def _get_targets(self, batch_gt_instances: InstanceData, batch_gt_instances_3d: InstanceData, feat_shape: tuple) -> Tuple[dict, int]:
        """
        Compute regression and classification targets in multiple images.

        Args:
            batch_gt_instances: InstanceList
            batch_gt_instances_3d: InstanceList
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            the dict ponents below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (B, wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (B, wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
               - offset_target_weight (Tensor): weights of and offset predict, shape (B, 2, H, W).
               - depth_target (Tensor): targets of depth predict, shape (B, 1, H, W).
               - depth_target_weight (Tensor): weights of and depth predict, shape (B, 1, H, W).
               - size3d_target (Tensor): targets of size3d predict, shape (B, 3, H, W).
               - size3d_target_weight (Tensor): weights of and size3d predict, shape (B, 3, H, W).
               - dir_bin_target (Tensor): targets of orientation predict, shape (B, num_bins * 2, H, W).
        """
        feat_h, feat_w = feat_shape
        
        with torch.no_grad():
            heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights, \
            size3d_target, size3d_target_weight, depth_target, depth_target_weight, dir_bin_target = multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                batch_gt_instances_3d,
                feat_shape=(feat_h, feat_w)
            )
        
        heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights = [
            torch.stack(t, dim=0).detach() for t in [
                heatmap_targets, 
                wh_targets, 
                wh_target_weights,
                offset_targets, 
                offset_target_weights
                ]]
        
        size3d_targets, size3d_target_weights, depth_targets, depth_target_weights, dir_bin_targets = [
            torch.stack(t, dim=0).detach() for t in [
                size3d_target, 
                size3d_target_weight, 
                depth_target,
                depth_target_weight, 
                dir_bin_target
                ]]

        target_result = dict(
            heatmap_targets=heatmap_targets,
            wh_targets=wh_targets,
            wh_target_weights=wh_target_weights,
            offset_targets=offset_targets,
            offset_target_weights=offset_target_weights,
            size3d_targets=size3d_targets,
            size3d_target_weights=size3d_target_weights,
            depth_targets=depth_targets,
            depth_target_weights=depth_target_weights,
            dir_bin_targets=dir_bin_targets
            )

        return target_result

    def _get_targets_single(self, gt_instances: InstanceData, gt_instances_3d: InstanceData, feat_shape: tuple):
        """
        Compute regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            gt_instances_3d (InstanceData):
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (2, H, W).
               - offset_target_weight (Tensor): weights of offset predict, shape (2, H, W).
               - depth_target (Tensor): targets of depth predict, shape (1, H, W).
               - depth_target_weight (Tensor): weights of and depth predict, shape (1, H, W).
               - size3d_target (Tensor): targets of size3d predict, shape (3, H, W).
               - size3d_target_weight (Tensor): weights of and size3d predict, shape (3, H, W).
               - dir_bin_target (Tensor): targets of orientation predict, shape (num_bins * 2, H, W).
        """
        feat_h, feat_w = feat_shape

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes3d = gt_instances_3d.bboxes_3d
        gt_size3d = gt_bboxes3d.tensor[:, 3:6]
        gt_depths = gt_instances_3d.depths

        # ecnode alpha
        gt_alpha = self.bbox_coder.encode(gt_bboxes3d)  # [n, self.num_dir_bins * 2]

        # init targets
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        wh_target = gt_bboxes.new_zeros((self.wh_dim, feat_h, feat_w))
        wh_target_weight = gt_bboxes.new_zeros((self.wh_dim, feat_h, feat_w))

        offset_target = gt_bboxes.new_zeros((2, feat_h, feat_w))
        offset_target_weight = gt_bboxes.new_zeros((2, feat_h, feat_w))

        depth_target = gt_bboxes.new_zeros((1, feat_h, feat_w))
        depth_target_weight = gt_bboxes.new_zeros((1, feat_h, feat_w))

        size3d_target = gt_bboxes.new_zeros((3, feat_h, feat_w))
        size3d_target_weight = gt_bboxes.new_zeros((3, feat_h, feat_w))

        dir_bin_target = gt_bboxes.new_zeros((self.num_dir_bins * 2, feat_h, feat_w))

        # down sample
        feat_bboxes = gt_bboxes.clone() / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        feat_bboxes_ctxs = (feat_bboxes[:, 0] + feat_bboxes[:, 2]) / 2
        feat_bboxes_ctxs_int = (feat_bboxes_ctxs).to(torch.int32)

        feat_bboxes_ctys = (feat_bboxes[:, 1] + feat_bboxes[:, 3]) / 2
        feat_bboxes_ctys_int = (feat_bboxes_ctys).to(torch.int32)

        for idx in range(feat_bboxes.shape[0]):
            obj_h, obj_w = feat_bboxes_hs[idx].item(), feat_bboxes_ws[idx].item()
            
            cat_id = gt_labels[idx]
            ctx, cty = feat_bboxes_ctxs[idx].item(), feat_bboxes_ctys[idx].item()
            ctx_int, cty_int = feat_bboxes_ctxs_int[idx].item(), feat_bboxes_ctys_int[idx].item()

            # heatmap
            radius = gaussian_radius([obj_h, obj_w],  min_overlap=0.3)
            radius = max(0, int(radius))
            gen_gaussian_target(heatmap_target[cat_id], [ctx_int, cty_int], radius)

            # wh
            if self.class_agnostic:
                wh_target[0, cty_int, ctx_int] = obj_w
                wh_target[1, cty_int, ctx_int] = obj_h
                wh_target_weight[:, cty_int, ctx_int] = 1
            else:
                s = int(cat_id * 2)
                wh_target[s, cty_int, ctx_int] = obj_w
                wh_target[s + 1, cty_int, ctx_int] = obj_h
                wh_target_weight[s: s + 2, cty_int, ctx_int] = 1

            # offset
            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int
            offset_target_weight[:, cty_int, ctx_int] = 1

            # depth
            depth_target[:, cty_int, ctx_int] = gt_depths[idx]
            depth_target_weight[:, cty_int, ctx_int] = 1

            # size3d
            size3d_target[:, cty_int, ctx_int] = gt_size3d[idx]  # [l, h, w]
            size3d_target_weight[:, cty_int, ctx_int] = 1

            # oirentations
            dir_bin_target[:, cty_int, ctx_int] = gt_alpha[idx]

        return heatmap_target, wh_target, wh_target_weight, offset_target, offset_target_weight, \
               size3d_target, size3d_target_weight, depth_target, depth_target_weight, dir_bin_target
               
    def predict_by_feat(self, 
                        center_heatmap_pred: Tensor, 
                        wh_pred: Tensor, 
                        offset_pred: Tensor, 
                        depth_pred: Tensor,
                        size3d_pred: Tensor,
                        dir_pred: Tensor,
                        batch_img_metas: Optional[List[dict]] = None, 
                        rescale: bool = True) -> InstanceList:
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info. Defaults to None.
            rescale (bool): If True, return boxes in original image space. Defaults to True.

        Returns:
            list[:obj:'InstanceData']: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - bboxes_3d (Tensor): Has a shape (num_instances, 7), the last dimension 4 arrange as (x, y, z, l, h, w, yaw).
                - scores_3d (Tensor): Classification scores, has a shape (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape (num_instances, ).
        """
        topk_bboxes2d, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins = self._decode(
            center_heatmap_pred, 
            wh_pred, 
            offset_pred, 
            depth_pred, 
            size3d_pred, 
            dir_pred)

        # 解析每个batch的结果
        result_list = []
        result3d_list = []
        for idx in range(topk_bboxes2d.shape[0]):
            scores = topk_scores[idx]
            keep = scores > self.score_thr

            img_meta = batch_img_metas[idx]

            if sum(keep):
                det_bboxes = topk_bboxes2d[idx][keep]

                # rescale -> raw
                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
                    
                det_scores = scores[keep]
                det_labels = topk_labels[idx][keep]

                cam2img = img_meta['cam2img']
                det_ctxs = ((det_bboxes[:, 0] + det_bboxes[:, 2]) / 2).unsqueeze(-1)   # [n, 1]
                det_ctys = ((det_bboxes[:, 1] + det_bboxes[:, 3]) / 2).unsqueeze(-1)   # [n, 1]
                det_locations = self.bbox_coder.decode_location(det_ctxs, det_ctys, topk_depth[idx][keep], cam2img)  # [n, 3]
                det_size3d = topk_size3d[idx][keep]   # [n, 3]

                # yaws
                det_dir_bins = topk_dir_bins[idx][keep]   # [n, num_bins * 4]
                det_alphas = self.bbox_coder.decode(det_dir_bins)   # [n, 1]
                det_yaws = self.bbox_coder.decode_orientation(det_alphas, det_locations)  # [n, 1]
              
                det_bboxes3d = torch.cat([det_locations, det_size3d, det_yaws], dim=-1)
            else:
                det_bboxes = topk_bboxes2d.new_zeros((0, 4))
                det_scores = topk_bboxes2d.new_zeros((0, ))
                det_labels = topk_bboxes2d.new_zeros((0, ))
                det_bboxes3d = topk_bboxes2d.new_zeros((0, 7))

            result = InstanceData()
            result.bboxes = det_bboxes
            result.scores = det_scores
            result.labels = det_labels
            result_list.append(result)

            result3d = InstanceData()
            result3d.bboxes_3d = img_meta['box_type_3d'](det_bboxes3d, box_dim=det_bboxes3d.shape[1], origin=(0.5, 0.5, 0.5))
            result3d.labels_3d = det_labels
            result3d.scores_3d = det_scores

            result3d_list.append(result3d)

        return result_list, result3d_list

    def _decode(self, center_heatmap_pred: Tensor, wh_pred: Tensor, offset_pred: Tensor, depth_pred: Tensor, size3d_pred: Tensor, dir_pred: Tensor):
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).

        Returns:
            Tuple[Tensor]: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - topk_bboxes (Tensor): Has a shape (B, topk, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - topk_scores (Tensor): Classification scores, has a shape (B, topk)
                - topk_labels (Tensor): Labels of bboxes, has a shape (B, topk).

                - topk_depth (Tensor): Depth of instance, has a shape (B, topk).
                - topk_size3d (Tensor): Size3d of instance, has a shape (B, topk, 3).
                - topk_dir_bins (Tensor): Orientation of instance, has a shape (B, topk, num_bins * 4). 
        """
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(center_heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == center_heatmap_pred).float()
        center_heatmap_pred = center_heatmap_pred * keep

        # topk
        batch_size, num_classes, output_h, output_w = center_heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(center_heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点偏移
        topk_offsets = transpose_and_gather_feat(offset_pred, topk_indexes)   # [n, topk, 2]
        topk_ctxs = topk_xs + topk_offsets[..., 0]
        topk_ctys = topk_ys + topk_offsets[..., 1]

        # wh
        topk_whs = transpose_and_gather_feat(wh_pred, topk_indexes)           # [n, topk, 2 if class_agnostic else 2 * num_classes]
        if not self.class_agnostic:
            topk_whs = topk_whs.view(batch_size, self.topk, num_classes, 2)
            classes_ind = topk_labels.view(batch_size, self.topk, 1, 1).expand(batch_size, self.topk, 1, 2).long()
            topk_whs = topk_whs.gather(2, classes_ind).view(batch_size, self.topk, 2)

        # bbox
        x1s = (topk_ctxs - topk_whs[..., 0] / 2)
        y1s = (topk_ctys - topk_whs[..., 1] / 2)
        x2s = (topk_ctxs + topk_whs[..., 0] / 2)
        y2s = (topk_ctys + topk_whs[..., 1] / 2)

        topk_bboxes = torch.stack([x1s, y1s, x2s, y2s], dim=2) * self.down_ratio

        # depth
        topk_depth = transpose_and_gather_feat(depth_pred, topk_indexes)    # [n, topk]
        topk_depth = 1. / topk_depth - 1

        # size3d
        topk_size3d = transpose_and_gather_feat(size3d_pred, topk_indexes)   # [n, topk, 3]

        # orietations
        topk_dir_bins = transpose_and_gather_feat(dir_pred, topk_indexes)    # [n, topk, self.num_dir_bins * 4]

        return topk_bboxes, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins
    

@MODELS.register_module()
class AlchemyCenterNetPlusMono3d(BaseMono3DDenseHead):
    """
    1. 使用location投影点坐标制作heatmap
    2. bbox回归点到边界框的距离
    3. 使用预测的投影点坐标计算location
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 down_ratio: int = 4,
                 stacked_convs: int = 2,
                 feat_channels: int = 128,
                 orientation_bin_margin: float = 0.3,
                 orientation_centers: List[float] = [0.5, -0.5],
                 loss_bbox: ConfigType = dict(type='EfficientIoULoss', loss_weight=5.0),
                 loss_depth: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_size3d: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_offset: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_orientation: ConfigType = dict(type='AlchemyMultiBinLoss', loss_weight=1.0),
                 loss_center_heatmap: ConfigType = dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 test_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_loc = None
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_dir_bins = len(orientation_centers)

        # post process
        self.topk = 100 if self.test_cfg is None else self.test_cfg.get('topk', 100)
        self.score_thr = 0.5 if self.test_cfg is None else self.test_cfg.get('score_thr', 0.5)        
        self.local_maximum_kernel = 3 if self.test_cfg is None else self.test_cfg.get('local_maximum_kernel', 3)

        # initial heads
        self._initial_heads()
        self.loss_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_depth = MODELS.build(loss_depth)
        self.loss_size3d = MODELS.build(loss_size3d)
        self.loss_dir = MODELS.build(loss_orientation)

        # coder
        self.bbox_coder = TASK_UTILS.build(dict(type='CenterNetMono3dCoder', num_dir_bins=self.num_dir_bins, bin_centers=orientation_centers, bin_margin=orientation_bin_margin))

        self.fp16_enabled = False
    
    def _initial_heads(self):
        """
        Initialize heads.
        """
        self.heatmap_head = self._build_head(self.num_classes)
        self.bbox_head = self._build_head(4)
        self.offset_head = self._build_head(2)
        self.depth_head = self._build_head(1)
        self.size3d_head = self._build_head(3)
        self.dir_head = self._build_head(self.num_dir_bins * 4)

    def _build_head(self, out_channels):
        """
        Build CenterNet head.

        Args:
            out_channels (int): The channels of output feature map.
        
        Return:
            head: (Sequential[module]): The module of head.
        """
        head_convs = []
        
        for i in range(self.stacked_convs):
            inp = self.in_channels if i == 0 else self.feat_channels
            head_convs.append(ConvModule(inp, self.feat_channels, 3, padding=1))

        head_convs.append(nn.Conv2d(self.feat_channels, out_channels, 1))

        return nn.Sequential(*head_convs)
    
    def init_weights(self) -> None:
        """
        Initialize weights of the head.
        """
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.bbox_head, self.offset_head, self.depth_head, self.size3d_head, self.dir_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
    
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        """
        Forward features.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            bbox_pred (Tensor): wh predicts, the channels number is wh_dim.
            offset_pred (Tensor): offset predicts, the channels number is 2.
            depth_pred (Tensor): depth predicts, the channels number is 1.
            size3d_pred (Tensor): size3d predicts, the channels number is 3.
            dir_pred (Tensor): orientation predicts, the channels number is num_bins * 4.
        """
        feat = x[0]
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        bbox_pred = self.bbox_head(feat).clamp(min=0)
        offset_pred = self.offset_head(feat)
        depth_pred = self.depth_head(feat).sigmoid()
        size3d_pred = self.size3d_head(feat)
        dir_pred = self.dir_head(feat)

        return (center_heatmap_pred, bbox_pred, offset_pred, depth_pred, size3d_pred, dir_pred)
    
    def loss_by_feat(self, 
                     center_heatmap_pred: Tensor, 
                     bbox_pred: Tensor, 
                     offset_pred: Tensor,
                     depth_pred: Tensor,
                     size3d_pred: Tensor,
                     dir_pred: Tensor,
                     batch_gt_instances_3d: InstanceList,
                     batch_gt_instances: InstanceList, 
                     batch_img_metas: List[dict], 
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        Compute losses of the head.

        Args:
            center_heatmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (list[Tensor]): wh predicts for all levels with shape (B, wh_dim, H, W).
            offset_pred (list[Tensor]): offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            batch_gt_instances (list[:obj:'InstanceData']): Batch of gt_instance. It usually includes ''bboxes'' and ''labels'' attributes.

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
                - loss_depth (Tensor): loss of depth heatmap.
                - loss_size3d (Tensor): loss of size3d heatmap.
                - loss_dir (Tensor): loss of orientation heatmap.
        """
        target_result = self._get_targets(batch_gt_instances, batch_gt_instances_3d, center_heatmap_pred.shape[2:])

        # heatmap loss
        center_heatmap_targets = target_result['heatmap_targets']
        center_heatmap_avg_factor = max(center_heatmap_targets.eq(1).sum(), 1)
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_heatmap(center_heatmap_pred, center_heatmap_targets, avg_factor=center_heatmap_avg_factor)

        # bbox loss
        H, W = center_heatmap_pred.shape[2:]
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step, dtype=torch.float32, device=center_heatmap_pred.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step, dtype=torch.float32, device=center_heatmap_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        bbox_preds = torch.cat((self.base_loc - bbox_pred[:, [0, 1]],
                                self.base_loc + bbox_pred[:, [2, 3]]), dim=1).permute(0, 2, 3, 1).reshape(-1, 4)  # [n*h*w, c]
        bbox_targets = target_result['bbox_targets'].permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target_weights = target_result['bbox_target_weights'].permute(0, 2, 3, 1).reshape(-1, 1)
        bbox_avg_factors = max(bbox_target_weights.eq(1).sum(), 1)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_target_weights.repeat(1, 4), avg_factor=bbox_avg_factors)

        # offset loss
        offset_targets = target_result['offset_targets']
        offset_target_weights = target_result['offset_target_weights']
        offset_avg_factors = max(offset_target_weights.eq(1).sum(), 1)
        loss_offset = self.loss_offset(offset_pred, offset_targets, offset_target_weights, avg_factor=offset_avg_factors)

        # depth loss
        depth_pred = 1 / depth_pred - 1.
        depth_targets = target_result['depth_targets']
        depth_target_weights = target_result['depth_target_weights']
        depth_avg_factors = max(depth_target_weights.eq(1).sum(), 1)
        loss_depth = self.loss_depth(depth_pred, depth_targets, depth_target_weights, avg_factor=depth_avg_factors)

        # size3d loss
        size3d_targets = target_result['size3d_targets']
        size3d_target_weights = target_result['size3d_target_weights']
        size3d_avg_factors = max(size3d_target_weights.eq(1).sum(), 1)
        loss_size3d = self.loss_size3d(size3d_pred, size3d_targets, size3d_target_weights, avg_factor=size3d_avg_factors)

        # oirentation loss
        dir_pred = dir_pred.permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 4)
        dir_bin_targets = target_result['dir_bin_targets'].permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 2)
        loss_dir = self.loss_dir(dir_pred, dir_bin_targets, self.num_dir_bins)

        return dict(
            loss_heatmap=loss_center_heatmap,
            loss_bbox=loss_bbox,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size3d=loss_size3d,
            loss_dir=loss_dir
            )

    def _get_targets(self, batch_gt_instances: InstanceData, batch_gt_instances_3d: InstanceData, feat_shape: tuple) -> Tuple[dict, int]:
        """
        Compute regression and classification targets in multiple images.

        Args:
            batch_gt_instances: InstanceList
            batch_gt_instances_3d: InstanceList
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            the dict ponents below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (B, wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (B, wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
               - offset_target_weight (Tensor): weights of and offset predict, shape (B, 2, H, W).
               - depth_target (Tensor): targets of depth predict, shape (B, 1, H, W).
               - depth_target_weight (Tensor): weights of and depth predict, shape (B, 1, H, W).
               - size3d_target (Tensor): targets of size3d predict, shape (B, 3, H, W).
               - size3d_target_weight (Tensor): weights of and size3d predict, shape (B, 3, H, W).
               - dir_bin_target (Tensor): targets of orientation predict, shape (B, num_bins * 2, H, W).
        """
        feat_h, feat_w = feat_shape
        
        with torch.no_grad():
            heatmap_targets, bbox_targets, bbox_target_weights, offset_targets, offset_target_weights, \
            size3d_target, size3d_target_weight, depth_target, depth_target_weight, dir_bin_target = multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                batch_gt_instances_3d,
                feat_shape=(feat_h, feat_w)
            )
        
        heatmap_targets, bbox_targets, bbox_target_weights, offset_targets, offset_target_weights = [
            torch.stack(t, dim=0).detach() for t in [
                heatmap_targets, 
                bbox_targets, 
                bbox_target_weights,
                offset_targets, 
                offset_target_weights
                ]]
        
        size3d_targets, size3d_target_weights, depth_targets, depth_target_weights, dir_bin_targets = [
            torch.stack(t, dim=0).detach() for t in [
                size3d_target, 
                size3d_target_weight, 
                depth_target,
                depth_target_weight, 
                dir_bin_target
                ]]

        target_result = dict(
            heatmap_targets=heatmap_targets,
            bbox_targets=bbox_targets,
            bbox_target_weights=bbox_target_weights,
            offset_targets=offset_targets,
            offset_target_weights=offset_target_weights,
            size3d_targets=size3d_targets,
            size3d_target_weights=size3d_target_weights,
            depth_targets=depth_targets,
            depth_target_weights=depth_target_weights,
            dir_bin_targets=dir_bin_targets
            )

        return target_result

    def _get_targets_single(self, gt_instances: InstanceData, gt_instances_3d: InstanceData, feat_shape: tuple):
        """
        Compute regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            gt_instances_3d (InstanceData):
            feat_shape (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (2, H, W).
               - offset_target_weight (Tensor): weights of offset predict, shape (2, H, W).
               - depth_target (Tensor): targets of depth predict, shape (1, H, W).
               - depth_target_weight (Tensor): weights of and depth predict, shape (1, H, W).
               - size3d_target (Tensor): targets of size3d predict, shape (3, H, W).
               - size3d_target_weight (Tensor): weights of and size3d predict, shape (3, H, W).
               - dir_bin_target (Tensor): targets of orientation predict, shape (num_bins * 2, H, W).
        """
        feat_h, feat_w = feat_shape

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes3d = gt_instances_3d.bboxes_3d
        gt_size3d = gt_bboxes3d.tensor[:, 3:6]
        gt_depths = gt_instances_3d.depths
        gt_centers_2d = gt_instances_3d.centers_2d  # cam坐标下的location投影到图像的坐标 [n, 2]

        # ecnode alpha
        gt_alpha = self.bbox_coder.encode(gt_bboxes3d)  # [n, self.num_dir_bins * 2]

        # init targets
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        bbox_target = gt_bboxes.new_ones((4, feat_h, feat_w)) * -1
        bbox_target_weight = gt_bboxes.new_zeros((1, feat_h, feat_w))

        offset_target = gt_bboxes.new_zeros((2, feat_h, feat_w))
        offset_target_weight = gt_bboxes.new_zeros((2, feat_h, feat_w))

        depth_target = gt_bboxes.new_zeros((1, feat_h, feat_w))
        depth_target_weight = gt_bboxes.new_zeros((1, feat_h, feat_w))

        size3d_target = gt_bboxes.new_zeros((3, feat_h, feat_w))
        size3d_target_weight = gt_bboxes.new_zeros((3, feat_h, feat_w))

        dir_bin_target = gt_bboxes.new_zeros((self.num_dir_bins * 2, feat_h, feat_w))

        # down sample
        feat_bboxes = gt_bboxes.clone() / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        feat_centers = gt_centers_2d.clone() / self.down_ratio
        feat_centers[:, 0] = torch.clamp(feat_centers[:, 0], min=0, max=feat_w - 1)
        feat_centers[:, 1] = torch.clamp(feat_centers[:, 1], min=0, max=feat_h - 1)
        feat_centers_int = (feat_centers).to(torch.int32)

        for idx in range(feat_centers.shape[0]):
            obj_h, obj_w = feat_bboxes_hs[idx].item(), feat_bboxes_ws[idx].item()

            cat_id = gt_labels[idx]

            ctx, cty = feat_centers[idx][0].item(), feat_centers[idx][1].item()
            ctx_int, cty_int = feat_centers_int[idx][0].item(), feat_centers_int[idx][1].item()

            # heatmap
            radius = gaussian_radius([obj_h, obj_w],  min_overlap=0.3)
            radius = max(0, int(radius))
            gen_gaussian_target(heatmap_target[cat_id], [ctx_int, cty_int], radius)

            # bbox
            bbox_target[:, cty_int, ctx_int] = gt_bboxes[idx]
            bbox_target_weight[:, cty_int, ctx_int] = 1

            # offset
            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int
            offset_target_weight[:, cty_int, ctx_int] = 1

            # depth
            depth_target[:, cty_int, ctx_int] = gt_depths[idx]
            depth_target_weight[:, cty_int, ctx_int] = 1

            # size3d
            size3d_target[:, cty_int, ctx_int] = gt_size3d[idx]  # [l, h, w]
            size3d_target_weight[:, cty_int, ctx_int] = 1

            # oirentations
            dir_bin_target[:, cty_int, ctx_int] = gt_alpha[idx]

        return heatmap_target, bbox_target, bbox_target_weight, offset_target, offset_target_weight, \
               size3d_target, size3d_target_weight, depth_target, depth_target_weight, dir_bin_target
               
    def predict_by_feat(self, 
                        center_heatmap_pred: Tensor, 
                        bbox_pred: Tensor, 
                        offset_pred: Tensor, 
                        depth_pred: Tensor,
                        size3d_pred: Tensor,
                        dir_pred: Tensor,
                        batch_img_metas: Optional[List[dict]] = None, 
                        rescale: bool = True) -> InstanceList:
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info. Defaults to None.
            rescale (bool): If True, return boxes in original image space. Defaults to True.

        Returns:
            list[:obj:'InstanceData']: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - bboxes_3d (Tensor): Has a shape (num_instances, 7), the last dimension 4 arrange as (x, y, z, l, h, w, yaw).
                - scores_3d (Tensor): Classification scores, has a shape (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape (num_instances, ).
        """
        topk_bboxes2d, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins, topk_proj_centers = self._decode(
            center_heatmap_pred, 
            bbox_pred, 
            offset_pred, 
            depth_pred, 
            size3d_pred, 
            dir_pred)

        # 解析每个batch的结果
        result_list = []
        result3d_list = []
        for idx in range(topk_bboxes2d.shape[0]):
            scores = topk_scores[idx]
            keep = scores > self.score_thr

            img_meta = batch_img_metas[idx]

            if sum(keep):
                det_bboxes = topk_bboxes2d[idx][keep]
                det_proj_centers = topk_proj_centers[idx][keep]

                # rescale -> raw
                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
                    det_proj_centers /= det_proj_centers.new_tensor(img_meta['scale_factor'])
                    
                det_scores = scores[keep]
                det_labels = topk_labels[idx][keep]

                cam2img = img_meta['cam2img']
                det_locations = self.bbox_coder.decode_location(det_proj_centers[:, 0], det_proj_centers[:, 1], topk_depth[idx][keep], cam2img)  # [n, 3]
                det_size3d = topk_size3d[idx][keep]   # [n, 3]

                # yaws
                det_dir_bins = topk_dir_bins[idx][keep]   # [n, num_bins * 4]
                det_alphas = self.bbox_coder.decode(det_dir_bins)   # [n, 1]
                det_yaws = self.bbox_coder.decode_orientation(det_alphas, det_locations)  # [n, 1]
              
                det_bboxes3d = torch.cat([det_locations, det_size3d, det_yaws], dim=-1)
            else:
                det_bboxes = topk_bboxes2d.new_zeros((0, 4))
                det_scores = topk_bboxes2d.new_zeros((0, ))
                det_labels = topk_bboxes2d.new_zeros((0, ))
                det_bboxes3d = topk_bboxes2d.new_zeros((0, 7))

            result = InstanceData()
            result.bboxes = det_bboxes
            result.scores = det_scores
            result.labels = det_labels
            result_list.append(result)

            result3d = InstanceData()
            result3d.bboxes_3d = img_meta['box_type_3d'](det_bboxes3d, box_dim=det_bboxes3d.shape[1], origin=(0.5, 0.5, 0.5))
            result3d.labels_3d = det_labels
            result3d.scores_3d = det_scores

            result3d_list.append(result3d)

        return result_list, result3d_list

    def _decode(self, center_heatmap_pred: Tensor, bbox_pred: Tensor, offset_pred: Tensor, depth_pred: Tensor, size3d_pred: Tensor, dir_pred: Tensor):
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).

        Returns:
            Tuple[Tensor]: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - topk_bboxes (Tensor): Has a shape (B, topk, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - topk_scores (Tensor): Classification scores, has a shape (B, topk)
                - topk_labels (Tensor): Labels of bboxes, has a shape (B, topk).

                - topk_depth (Tensor): Depth of instance, has a shape (B, topk).
                - topk_size3d (Tensor): Size3d of instance, has a shape (B, topk, 3).
                - topk_dir_bins (Tensor): Orientation of instance, has a shape (B, topk, num_bins * 4). 
                - topk_proj_centers (Tensor): Project center of instance, has a shape (B, topk, 2). 
        """
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(center_heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == center_heatmap_pred).float()
        center_heatmap_pred = center_heatmap_pred * keep

        # topk
        batch_size, _, output_h, output_w = center_heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(center_heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点偏移
        topk_offsets = transpose_and_gather_feat(offset_pred, topk_indexes)   # [n, topk, 2]
        topk_ctxs = (topk_xs + topk_offsets[..., 0]) * self.down_ratio
        topk_ctys = (topk_ys + topk_offsets[..., 1]) * self.down_ratio

        # wh
        topk_whs = transpose_and_gather_feat(bbox_pred, topk_indexes)          # [n, topk, 4]

        # bbox
        x1s = topk_ctxs - topk_whs[..., 0]
        y1s = topk_ctys - topk_whs[..., 1]
        x2s = topk_ctxs + topk_whs[..., 2]
        y2s = topk_ctys + topk_whs[..., 3]
        topk_bboxes = torch.stack([x1s, y1s, x2s, y2s], dim=2)

        # depth
        topk_depth = transpose_and_gather_feat(depth_pred, topk_indexes)    # [n, topk]
        topk_depth = 1. / topk_depth - 1

        # size3d
        topk_size3d = transpose_and_gather_feat(size3d_pred, topk_indexes)   # [n, topk, 3]

        # orietations
        topk_dir_bins = transpose_and_gather_feat(dir_pred, topk_indexes)    # [n, topk, self.num_dir_bins * 4]

        # topk proj centers
        topk_proj_centers = torch.cat([topk_ctxs.unsqueeze(-1), topk_ctys.unsqueeze(-1)], dim=-1)  # [n, topk, 2]

        return topk_bboxes, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins, topk_proj_centers


@MODELS.register_module()
class AlchemyCenterNetPlusV2Mono3d(AlchemyCenterNetPlusMono3d):
    """
    1. 使用location投影点坐标制作heatmap
    2. bbox回归点到边界框的距离
    3. 使用预测的投影点坐标计算location
    4. 预测depth_bias, depth = 1 / sigmoid(x) + depth_bias
    """
    def _initial_heads(self):
        self.heatmap_head = self._build_head(self.num_classes)
        self.bbox_head = self._build_head(4)
        self.offset_head = self._build_head(2)
        self.depth_head = self._build_head(1)
        self.size3d_head = self._build_head(3)
        self.dir_head = self._build_head(self.num_dir_bins * 4)
        self.depth_bias_head = self._build_head(1)
    
    def init_weights(self) -> None:
        """
        Initialize weights of the head.
        """
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.bbox_head, self.offset_head, self.depth_head, self.size3d_head, self.dir_head, self.depth_bias_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
    
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        """
        Forward features.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            bbox_pred (Tensor): wh predicts, the channels number is wh_dim.
            offset_pred (Tensor): offset predicts, the channels number is 2.
            depth_pred (Tensor): depth predicts, the channels number is 1.
            depth_bias_pred (Tensor): depth bias predicts, the channels number is 1.
            size3d_pred (Tensor): size3d predicts, the channels number is 3.
            dir_pred (Tensor): orientation predicts, the channels number is num_bins * 4.
        """
        feat = x[0]
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        bbox_pred = self.bbox_head(feat).clamp(min=0)
        offset_pred = self.offset_head(feat)
        depth_pred = self.depth_head(feat).sigmoid()
        size3d_pred = self.size3d_head(feat)
        dir_pred = self.dir_head(feat)
        depth_bias_pred = self.depth_bias_head(feat)

        return (center_heatmap_pred, bbox_pred, offset_pred, depth_pred, size3d_pred, dir_pred, depth_bias_pred)
    
    def loss_by_feat(self, 
                     center_heatmap_pred: Tensor, 
                     bbox_pred: Tensor, 
                     offset_pred: Tensor,
                     depth_pred: Tensor,
                     size3d_pred: Tensor,
                     dir_pred: Tensor,
                     depth_bias_pred: Tensor,
                     batch_gt_instances_3d: InstanceList,
                     batch_gt_instances: InstanceList, 
                     batch_img_metas: List[dict], 
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        Compute losses of the head.

        Args:
            center_heatmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (list[Tensor]): wh predicts for all levels with shape (B, wh_dim, H, W).
            offset_pred (list[Tensor]): offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            depth_bias_pred (Tensor): depth bias predicts with shape (B, 1, H, W).
            batch_gt_instances (list[:obj:'InstanceData']): Batch of gt_instance. It usually includes ''bboxes'' and ''labels'' attributes.

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
                - loss_depth (Tensor): loss of depth heatmap.
                - loss_size3d (Tensor): loss of size3d heatmap.
                - loss_dir (Tensor): loss of orientation heatmap.
        """
        target_result = self._get_targets(batch_gt_instances, batch_gt_instances_3d, center_heatmap_pred.shape[2:])

        # heatmap loss
        center_heatmap_targets = target_result['heatmap_targets']
        center_heatmap_avg_factor = max(center_heatmap_targets.eq(1).sum(), 1)
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_heatmap(center_heatmap_pred, center_heatmap_targets, avg_factor=center_heatmap_avg_factor)

        # bbox loss
        H, W = center_heatmap_pred.shape[2:]
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step, dtype=torch.float32, device=center_heatmap_pred.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step, dtype=torch.float32, device=center_heatmap_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        bbox_preds = torch.cat((self.base_loc - bbox_pred[:, [0, 1]],
                                self.base_loc + bbox_pred[:, [2, 3]]), dim=1).permute(0, 2, 3, 1).reshape(-1, 4)  # [n*h*w, c]
        bbox_targets = target_result['bbox_targets'].permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target_weights = target_result['bbox_target_weights'].permute(0, 2, 3, 1).reshape(-1, 1)
        bbox_avg_factors = max(bbox_target_weights.eq(1).sum(), 1)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_target_weights.repeat(1, 4), avg_factor=bbox_avg_factors)

        # offset loss
        offset_targets = target_result['offset_targets']
        offset_target_weights = target_result['offset_target_weights']
        offset_avg_factors = max(offset_target_weights.eq(1).sum(), 1)
        loss_offset = self.loss_offset(offset_pred, offset_targets, offset_target_weights, avg_factor=offset_avg_factors)

        # depth loss
        depth_pred = 1 / depth_pred + depth_bias_pred
        depth_targets = target_result['depth_targets']
        depth_target_weights = target_result['depth_target_weights']
        depth_avg_factors = max(depth_target_weights.eq(1).sum(), 1)
        loss_depth = self.loss_depth(depth_pred, depth_targets, depth_target_weights, avg_factor=depth_avg_factors)

        # size3d loss
        size3d_targets = target_result['size3d_targets']
        size3d_target_weights = target_result['size3d_target_weights']
        size3d_avg_factors = max(size3d_target_weights.eq(1).sum(), 1)
        loss_size3d = self.loss_size3d(size3d_pred, size3d_targets, size3d_target_weights, avg_factor=size3d_avg_factors)

        # oirentation loss
        dir_pred = dir_pred.permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 4)
        dir_bin_targets = target_result['dir_bin_targets'].permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins * 2)
        loss_dir = self.loss_dir(dir_pred, dir_bin_targets, self.num_dir_bins)

        return dict(
            loss_heatmap=loss_center_heatmap,
            loss_bbox=loss_bbox,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size3d=loss_size3d,
            loss_dir=loss_dir
            )
               
    def predict_by_feat(self, 
                        center_heatmap_pred: Tensor, 
                        bbox_pred: Tensor, 
                        offset_pred: Tensor, 
                        depth_pred: Tensor,
                        size3d_pred: Tensor,
                        dir_pred: Tensor,
                        depth_bias_pred: Tensor,
                        batch_img_metas: Optional[List[dict]] = None, 
                        rescale: bool = True) -> InstanceList:
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).
            depth_bias_pred (Tensor): depth bias predicts with shape (B, 1, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info. Defaults to None.
            rescale (bool): If True, return boxes in original image space. Defaults to True.

        Returns:
            list[:obj:'InstanceData']: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - bboxes_3d (Tensor): Has a shape (num_instances, 7), the last dimension 4 arrange as (x, y, z, l, h, w, yaw).
                - scores_3d (Tensor): Classification scores, has a shape (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape (num_instances, ).
        """
        topk_bboxes2d, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins, topk_proj_centers = self._decode(
            center_heatmap_pred, 
            bbox_pred, 
            offset_pred, 
            depth_pred, 
            size3d_pred, 
            dir_pred,
            depth_bias_pred
            )

        # 解析每个batch的结果
        result_list = []
        result3d_list = []
        for idx in range(topk_bboxes2d.shape[0]):
            scores = topk_scores[idx]
            keep = scores > self.score_thr

            img_meta = batch_img_metas[idx]

            if sum(keep):
                det_bboxes = topk_bboxes2d[idx][keep]
                det_proj_centers = topk_proj_centers[idx][keep]

                # rescale -> raw
                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
                    det_proj_centers /= det_proj_centers.new_tensor(img_meta['scale_factor'])
                    
                det_scores = scores[keep]
                det_labels = topk_labels[idx][keep]

                cam2img = img_meta['cam2img']
                det_locations = self.bbox_coder.decode_location(det_proj_centers[:, 0], det_proj_centers[:, 1], topk_depth[idx][keep], cam2img)  # [n, 3]
                det_size3d = topk_size3d[idx][keep]   # [n, 3]

                # yaws
                det_dir_bins = topk_dir_bins[idx][keep]   # [n, num_bins * 4]
                det_alphas = self.bbox_coder.decode(det_dir_bins)   # [n, 1]
                det_yaws = self.bbox_coder.decode_orientation(det_alphas, det_locations)  # [n, 1]
              
                det_bboxes3d = torch.cat([det_locations, det_size3d, det_yaws], dim=-1)
            else:
                det_bboxes = topk_bboxes2d.new_zeros((0, 4))
                det_scores = topk_bboxes2d.new_zeros((0, ))
                det_labels = topk_bboxes2d.new_zeros((0, ))
                det_bboxes3d = topk_bboxes2d.new_zeros((0, 7))

            result = InstanceData()
            result.bboxes = det_bboxes
            result.scores = det_scores
            result.labels = det_labels
            result_list.append(result)

            result3d = InstanceData()
            result3d.bboxes_3d = img_meta['box_type_3d'](det_bboxes3d, box_dim=det_bboxes3d.shape[1], origin=(0.5, 0.5, 0.5))
            result3d.labels_3d = det_labels
            result3d.scores_3d = det_scores

            result3d_list.append(result3d)

        return result_list, result3d_list

    def _decode(self, center_heatmap_pred: Tensor, bbox_pred: Tensor, offset_pred: Tensor, depth_pred: Tensor, size3d_pred: Tensor, dir_pred: Tensor, depth_bias_pred: Tensor):
        """
        Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (Tensor): WH predicts with shape (B, wh_dim, H, W).
            offset_pred (Tensor): Offset predicts with shape (B, 2, H, W).
            depth_pred (Tensor): Depth predicts with shape (B, 1, H, W).
            size3d_pred (Tensor): Size3d predicts with shape (B, 3, H, W).
            dir_pred (Tensor): orientation predicts with shape (B, num_bins * 4, H, W).

        Returns:
            Tuple[Tensor]: Instance segmentation results of each image after the post process.
            Each item usually contains following keys.
                - topk_bboxes (Tensor): Has a shape (B, topk, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
                - topk_scores (Tensor): Classification scores, has a shape (B, topk)
                - topk_labels (Tensor): Labels of bboxes, has a shape (B, topk).

                - topk_depth (Tensor): Depth of instance, has a shape (B, topk).
                - topk_size3d (Tensor): Size3d of instance, has a shape (B, topk, 3).
                - topk_dir_bins (Tensor): Orientation of instance, has a shape (B, topk, num_bins * 4). 
                - topk_proj_centers (Tensor): Project center of instance, has a shape (B, topk, 2). 
        """
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(center_heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == center_heatmap_pred).float()
        center_heatmap_pred = center_heatmap_pred * keep

        # topk
        batch_size, _, output_h, output_w = center_heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(center_heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点偏移
        topk_offsets = transpose_and_gather_feat(offset_pred, topk_indexes)   # [n, topk, 2]
        topk_ctxs = (topk_xs + topk_offsets[..., 0]) * self.down_ratio
        topk_ctys = (topk_ys + topk_offsets[..., 1]) * self.down_ratio

        # wh
        topk_whs = transpose_and_gather_feat(bbox_pred, topk_indexes)          # [n, topk, 4]

        # bbox
        x1s = topk_ctxs - topk_whs[..., 0]
        y1s = topk_ctys - topk_whs[..., 1]
        x2s = topk_ctxs + topk_whs[..., 2]
        y2s = topk_ctys + topk_whs[..., 3]
        topk_bboxes = torch.stack([x1s, y1s, x2s, y2s], dim=2)

        # depth
        topk_depth = transpose_and_gather_feat(depth_pred, topk_indexes)    # [n, topk]
        topk_depth_bias = transpose_and_gather_feat(depth_bias_pred, topk_indexes)
        topk_depth = 1. / topk_depth + topk_depth_bias

        # size3d
        topk_size3d = transpose_and_gather_feat(size3d_pred, topk_indexes)   # [n, topk, 3]

        # orietations
        topk_dir_bins = transpose_and_gather_feat(dir_pred, topk_indexes)    # [n, topk, self.num_dir_bins * 4]

        # topk proj centers
        topk_proj_centers = torch.cat([topk_ctxs.unsqueeze(-1), topk_ctys.unsqueeze(-1)], dim=-1)  # [n, topk, 2]

        return topk_bboxes, topk_scores, topk_labels, topk_depth, topk_size3d, topk_dir_bins, topk_proj_centers

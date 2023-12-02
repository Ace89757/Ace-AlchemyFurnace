# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import torch
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from typing import List, Tuple, Union

from mmdet.models.detectors.base import BaseDetector
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from alchemy.registry import MODELS


@MODELS.register_module()
class AlchemyDet2dDetector(BaseDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: List[OptConfigType] = None,
                 rpn: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # stem
        self._build_stem(backbone, neck)

        # rpn
        if rpn is not None:
            self._build_rpn(rpn)
        
        # head
        self._build_head(head)
    
    def _build_stem(self, backbone, neck):
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            if isinstance(neck, dict):
                self.neck = MODELS.build(neck)
                
            elif isinstance(neck, list):
                necks = []
                for neck_cfg in neck:
                    necks.append(MODELS.build(neck_cfg))
                
                self.neck = nn.Sequential(*necks)
    
    def _build_rpn(self, rpn):
        if self.train_cfg is not None and 'rpn' in self.train_cfg:
            rpn.update(train_cfg=self.train_cfg['rpn'])

        if self.test_cfg is not None and 'rpn' in self.test_cfg:
            rpn.update(test_cfg=self.test_cfg['rpn'])
        
        rpn.update(num_classes=1)

        self.rpn = MODELS.build(rpn)

    def _build_head(self, head):
        if self.train_cfg is not None and 'head' in self.train_cfg:
            head.update(train_cfg=self.train_cfg['head'])

        if self.test_cfg is not None and 'head' in self.test_cfg:
            head.update(test_cfg=self.test_cfg['head'])
            
        self.head = MODELS.build(head)

    @property
    def has_rpn(self) -> bool:
        return hasattr(self, 'rpn') and self.rpn is not None

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """
        Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W). These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:'DetDataSample']): The batch data samples. 
                                                             It usually includes information such as 'gt_instance' or 'gt_panoptic_seg' or 'gt_sem_seg'.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # rpn
        if self.has_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = deepcopy(batch_data_samples)
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn.loss_and_predict(x, rpn_data_samples, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            head_inputs = (x, rpn_results_list, batch_data_samples)
        else:
            head_inputs = (x, batch_data_samples)
        
        # head
        head_losses = self.head.loss(*head_inputs)
        losses.update(head_losses)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        """
        Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:'DetDataSample']): The Data Samples. It usually includes information such as 'gt_instance', 'gt_panoptic_seg' and 'gt_sem_seg'.
            rescale (bool): Whether to rescale the results. Defaults to True.

        Returns:
            list[:obj:'DetDataSample']: Detection results of the
            input images. Each DetDataSample usually contain 'pred_instances'. And the 'pred_instances' usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)

        # rpn
        if self.has_rpn:
            # If there are no pre-defined proposals, use RPN to get proposals
            if batch_data_samples[0].get('proposals', None) is None:
                rpn_results_list = self.rpn.predict(x, batch_data_samples, rescale=False)
            else:
                rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

            head_inputs = (x, rpn_results_list, batch_data_samples)
        else:
            head_inputs = (x, batch_data_samples)

        # head
        results_list = self.head.predict(*head_inputs, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """
        Network forward process. Usually includes backbone, neck and head forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:'DetDataSample']): Each item contains the meta information of each image and corresponding annotations.

        Returns:
            tuple[list]: A tuple of features from 'bbox_head' forward.
        """
        x = self.extract_feat(batch_inputs)

        # rpn
        if self.has_rpn:
            rpn_results_list = self.rpn.predict(x, batch_data_samples, rescale=False)
            head_inputs = (x, rpn_results_list, batch_data_samples)
        else:
            head_inputs = (x, )

        # head
        results = self.head.forward(*head_inputs)

        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """
        Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different resolutions.
        """
        x = self.backbone(batch_inputs)
        
        if self.with_neck:
            x = self.neck(x)

        return x

# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

from torch import Tensor
from typing import Tuple

from mmdet.structures import SampleList
from mmdet3d.utils import OptInstanceList
from mmengine.structures import InstanceData

from alchemy.registry import MODELS
from alchemy.models.detectors.det2d_detector import AlchemyDet2dDetector


@MODELS.register_module()
class AlchemyMono3dDetector(AlchemyDet2dDetector):
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """
        Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different resolutions.
        """
        batch_imgs = batch_inputs['imgs']

        x = self.backbone(batch_imgs)
        if self.with_neck:
            x = self.neck(x)

        return x

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
        results2d_list, results3d_list = self.head.predict(*head_inputs, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results2d_list, results3d_list)

        return batch_data_samples
    
    def add_pred_to_datasample(self, data_samples: SampleList, data_instances_2d: OptInstanceList = None, data_instances_3d: OptInstanceList = None) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D Detection results of each image. Defaults to None.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D Detection results of each image. Defaults to None.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are 2D prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape (num_instances, 4).
        """

        assert (data_instances_2d is not None) or (data_instances_3d is not None), 'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [InstanceData() for _ in range(len(data_instances_3d))]

        if data_instances_3d is None:
            data_instances_3d = [InstanceData() for _ in range(len(data_instances_2d))]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
            
        return data_samples
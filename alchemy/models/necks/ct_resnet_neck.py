# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.

import math
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from typing import Sequence, Tuple
from mmengine.model import BaseModule
from mmdet.utils import OptMultiConfig

from alchemy.registry import MODELS


@MODELS.register_module()
class AlchemyCTResNetNeck(BaseModule):
    """
    The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for object classification and box regression.

    Args:
         in_channels (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Defaults to True.
         init_cfg (:obj:`ConfigDict` or dict or list[dict] or list[:obj:`ConfigDict`], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 num_deconv_filters: Tuple[int, ...],
                 num_deconv_kernels: Tuple[int, ...],
                 use_dcn: bool = True,
                 upsample_mode: str = 'deconv',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)

        self.use_dcn = use_dcn
        self.in_channels = in_channels
        self.upsample_mode = upsample_mode
        self.num_deconv_filters = num_deconv_filters

        self.deconv_layers = self._make_deconv_layer(num_deconv_filters, num_deconv_kernels)

        self.fp16_enabled = False

    def _make_deconv_layer(self, 
                           num_deconv_filters: Tuple[int, ...], 
                           num_deconv_kernels: Tuple[int, ...]) -> nn.Sequential:
        """
        use deconv layers to upsample backbone's output.
        """
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channels = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channels,
                feat_channels,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            
            layers.append(conv_module)

            if self.upsample_mode == 'nearest':
                upsample_module = nn.Upsample(scale_factor=num_deconv_kernels[i] // 2, mode='nearest')
                smooth_module = ConvModule(
                    feat_channels,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'))
                
                layers.append(upsample_module)
                layers.append(smooth_module)
                
            elif self.upsample_mode == 'bilinear':
                upsample_module = nn.Upsample(scale_factor=num_deconv_kernels[i] // 2, mode='bilinear', align_corners=False)
                smooth_module = ConvModule(
                    feat_channels,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'))
                
                layers.append(upsample_module)
                layers.append(smooth_module)
            else:
                upsample_module = ConvModule(
                    feat_channels,
                    feat_channels,
                    num_deconv_kernels[i],
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='deconv'),
                    norm_cfg=dict(type='BN'))
            
                layers.append(upsample_module)

            self.in_channels = feat_channels

        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """
        Initialize the parameters.
        """
        # for m in self.modules():
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code, reset the ConvTranspose2d initialization parameters
                m.reset_parameters()

                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code, reset the Conv2d initialization parameters
                m.reset_parameters()

            if self.upsample_mode in ['nearest', 'bilinear']:
                if name in [f'deconv_layers.{idx * 3 + 2}.conv' for idx in range(len(self.num_deconv_filters))]:
                    m.reset_parameters()

                    # Simulated bilinear upsampling kernel
                    w = m.weight.data
                    f = math.ceil(w.size(2) / 2)
                    c = (2 * f - 1 - f % 2) / (2. * f)
                    for i in range(w.size(2)):
                        for j in range(w.size(3)):
                            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

                    for c in range(1, w.size(0)):
                        w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        """model forward."""
        assert isinstance(x, (list, tuple))
        outs = self.deconv_layers(x[-1])
        return outs,

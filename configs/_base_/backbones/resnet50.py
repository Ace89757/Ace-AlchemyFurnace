# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    norm_eval=False,
    frozen_stages=-1,
    out_indices=(0, 1, 2, 3),
    norm_cfg=dict(type='BN', requires_grad=True),
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
)
_base_ = [
    '../../../_base_/backbones/resnet50.py',
    '../../../_base_/datasets/kitti_mono3d.py',
    '../../../_base_/schedules/schedule_3x.py',
    '../../../_base_/runtimes/runtime_mmdet3d.py'
]

batch_size = 4

"""
model
"""
model = dict(
    type='AlchemyMono3dDetector',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck=[
        dict(
            type='mmdet.DilatedEncoder',
            in_channels=2048,
            out_channels=2048,
            block_mid_channels=512,
            num_residual_blocks=4,
            block_dilations=(2, 4, 6, 8)
        ),
        dict(
            type='AlchemyCTResNetNeck',
            in_channels=2048,
            num_deconv_filters=(1024, 512, 256),
            num_deconv_kernels=(4, 4, 4),
            use_dcn=False,
            upsample_mode='nearest'
        )
    ],
    head=dict(
        type='AlchemyCenterNetMono3d', 
        in_channels=256, 
        feat_channels=256,
        orientation_centers=[
            -1 / 4, 
            -3 / 4, 
            1 / 4,
            3 / 4
            ],
        orientation_bin_margin=1 / 18,
        num_classes={{_base_.num_classes}}),
    train_cfg=None,
    test_cfg=dict(topk=100, max_per_img=100, local_maximum_kernel=3, score_thr={{_base_.score_thr}}))


"""
dataloader
"""
train_dataloader = dict(
    batch_size=batch_size, 
    num_workers={{_base_.num_workers}}
    )

val_dataloader = dict(
    batch_size=batch_size,
    num_workers={{_base_.num_workers}}
    )

test_dataloader = val_dataloader


"""
optimizer
"""

optim_wrapper = dict(
    optimizer=dict(lr=1.25e-4),
    )


param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=200)
]

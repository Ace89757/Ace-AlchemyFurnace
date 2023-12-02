_base_ = [
    '../../../_base_/backbones/resnet50.py',
    '../../../_base_/datasets/bdd100k_2d.py',
    '../../../_base_/schedules/schedule_2x.py',
    '../../../_base_/runtimes/runtime_mmdet.py'
]

batch_size = 8

"""
model
"""

model = dict(
    type='AlchemyDet2dDetector',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck=dict(
        type='AlchemyCTResNetNeck',
        in_channels=2048,
        num_deconv_filters=(1024, 512, 256),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=False,
        upsample_mode='nearest'
        ),
    head=dict(
        type='AlchemyCenterNet',
        stacked_convs=2,
        in_channels=256,
        feat_channels=256,
        class_agnostic=True, 
        num_classes={{_base_.num_classes}}),
    test_cfg=dict(head=dict(topk=100, max_per_img=100, local_maximum_kernel=3, score_thr={{_base_.score_thr}})
    ))


"""
dataloader
"""
train_dataloader = dict(
    batch_size=batch_size,
    num_workers={{_base_.num_workers}},
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
    clip_grad=dict(max_norm=35, norm_type=2)
    )


auto_scale_lr = dict(base_batch_size=32)
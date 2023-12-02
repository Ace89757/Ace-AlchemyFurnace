_base_ = [
    '../centernet/ct_det2d_r50_24e_v3.py'
]

"""
model
"""

model = dict(
    head=dict(
        _delete_=True,
        type='AlchemyTTFNet',
        alpha=0.54,
        bbox_convs=2,
        heatmap_convs=2,
        in_channels=256,
        bbox_channels=128,
        heatmap_channels=256,
        num_classes={{_base_.num_classes}},
        loss_bbox=dict(type='EfficientIoULoss', loss_weight=5.0),
        loss_center_heatmap=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
        ),
    )

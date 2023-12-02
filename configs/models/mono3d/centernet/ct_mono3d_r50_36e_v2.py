_base_ = [
    './ct_mono3d_r50_36e_v1.py'
]


"""
model
"""

model = dict(
    head=dict(
        _delete_=True,
        type='AlchemyCenterNetPlusMono3d',
        in_channels=256, 
        feat_channels=256,
        orientation_centers=[
            -1 / 4, 
            -3 / 4, 
            1 / 4,
            3 / 4
            ],
        orientation_bin_margin=1 / 18,
        num_classes={{_base_.num_classes}})
    )

_base_ = [
    './ct_det2d_r50_24e_v1.py'
]

"""
model
"""

model = dict(
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
    ]
    )

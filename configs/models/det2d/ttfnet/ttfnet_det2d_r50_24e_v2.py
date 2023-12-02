_base_ = [
    './ttfnet_det2d_r50_24e_v1.py'
]

"""
model
"""

model = dict(
    head=dict(
        type='AlchemyTTFNetPlus'
        ),
    )

_base_ = [
    './ct_det2d_r50_24e_v1.py'
]

"""
model
"""

model = dict(
    neck=dict(
        use_dcn=True
        )
    )

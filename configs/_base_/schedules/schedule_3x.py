"""
training schedule for 3x
"""
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


"""
learning rate
"""
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[24, 33], gamma=0.1)
]


"""
optimizer
"""
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)

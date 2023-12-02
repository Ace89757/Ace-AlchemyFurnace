default_scope = 'alchemy'

num_workers = 8
log_interval = 10
val_interval = 1
ckpt_interval = 1

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=log_interval),             # 打印日志
    param_scheduler=dict(type='ParamSchedulerHook'),                   # 调用 ParamScheduler 的 step 方法
    checkpoint=dict(type='CheckpointHook', interval=ckpt_interval),    # 按指定间隔保存权重
    sampler_seed=dict(type='DistSamplerSeedHook'),                     # 在分布式训练时调用 Sampler 的 step 方法以确保 shuffle 参数生效
    visualization=dict(type='mmdet3d.Det3DVisualizationHook'),
    )

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=3407)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    ]

visualizer = dict(type='mmdet3d.Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=log_interval, by_epoch=True, num_digits=3, log_with_hierarchy=True)

log_level = 'INFO'
load_from = None
resume = False

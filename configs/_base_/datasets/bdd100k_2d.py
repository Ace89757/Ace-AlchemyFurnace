"""
dataset settings
"""
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/bdd100k/'

train_ann_file = 'annotations/bdd100k_train.json'
val_ann_file = 'annotations/bdd100k_val.json'
test_ann_file = 'annotations/bdd100k_val.json'

metainfo = {
        'classes': ('Car', 'Bus', 'Truck', 'Person', 'Rider', 'Motor', 'Bike', 'Sign', 'Light', 'Other'),
        'palette': [(238, 180, 34), (205, 205, 0), (102, 205, 0), (141, 238, 238),  (67, 205, 128), (0, 206, 209), (188, 208, 104), (178, 34, 34), (138, 43, 226), (46, 139, 87)]
    }

num_classes = len(metainfo['classes'])

"""
args
"""
score_thr = 0.5
backend_args = None


"""
data preprocessor
"""
data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor', 
    mean=[71.072, 74.628, 73.946],   # rgb 
    std=[50.318, 50.5, 51.186],    # rgb 
    bgr_to_rgb=True,
    pad_size_divisor=32
    )


"""
evaluator
"""
val_evaluator = dict(
    type='AlchemyDet2dMetric',
    ann_file=data_root + val_ann_file,
    max_dets=(10, 30, 100),   # 每张图片, 按置信度排序, 排名前10、前30、前100个预测框的指标
    object_size=(32, 64, 1e5),     # 表示小、中、大目标的边长
    format_only=False
    )

test_evaluator = val_evaluator


"""
pipeline
"""
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=(608, ), keep_ratio=True),   # 将最大边resize到指定尺寸
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=(608, ), keep_ratio=True),   # 将最大边resize到指定尺寸
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]



"""
dataloader
"""
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_ann_file,
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            metainfo=metainfo
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo
        )
    )

test_dataloader = val_dataloader

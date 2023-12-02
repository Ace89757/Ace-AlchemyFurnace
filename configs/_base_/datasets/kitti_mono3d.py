"""
dataset settings
"""
dataset_type = 'mmdet3d.KittiDataset'
data_root = 'data/kitti/'

train_ann_file = 'kitti_infos_train.pkl'
val_ann_file = 'kitti_infos_val.pkl'
test_ann_file = 'kitti_infos_val.pkl'

metainfo = {
    'classes': ('Pedestrian', 'Cyclist', 'Car'),
    'palette': [(238, 180, 34), (205, 205, 0), (102, 205, 0)]
    }

input_modality = dict(use_lidar=False, use_camera=True)

num_classes = len(metainfo['classes'])

"""
args
"""
score_thr = 0.25
backend_args = None


"""
data preprocessor
"""
data_preprocessor=dict(
    type='mmdet3d.Det3DDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32
    )


"""
evaluator
"""
val_evaluator = dict(
    type='AlchemyMono3dMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    collect_device='gpu',
    backend_args=backend_args
    )

test_evaluator = val_evaluator


"""
transforms
"""

train_pipeline = [
    dict(type='mmdet3d.LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet3d.LoadAnnotations3D', 
         with_bbox=True,
         with_label=True,
         with_attr_label=False,
         with_bbox_3d=True,
         with_label_3d=True,
         with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet3d.Pack3DDetInputs',
         keys=['img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths']),
]

test_pipeline = [
    dict(type='mmdet3d.LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='mmdet3d.Pack3DDetInputs', keys=['img'])
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
        data_prefix=dict(img='training/image_2'),
        load_type='fov_image_based',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        box_type_3d='Camera',
        test_mode=False,
        backend_args=backend_args
    ))


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='training/image_2'),
        ann_file=val_ann_file,
        load_type='fov_image_based',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Camera',
        backend_args=backend_args))

test_dataloader = val_dataloader
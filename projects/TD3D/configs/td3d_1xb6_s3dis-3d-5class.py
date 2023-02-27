_base_ = ['./td3d.py', 'mmdet3d::_base_/datasets/s3dis-3d.py']
custom_imports = dict(imports=['projects.TD3D.td3d'])

# model settings
model = dict(bbox_head=dict(num_classes=5))

# dataset settings
dataset_type = 'S3DISDataset'
data_root = 'data/s3dis/'
metainfo = dict(classes=('table', 'chair', 'sofa', 'bookcase', 'board'))
train_area = [1, 2, 3, 4, 6]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='PointSample', num_points=100000),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],  # todo: do we need it?
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        dataset=dict(datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f's3dis_infos_Area_{i}.pkl',
                pipeline=train_pipeline,
                filter_empty_gt=True,
                metainfo=metainfo,
                box_type_3d='Depth') for i in train_area
        ])))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='TD3DInstanceSegMetric')
test_evaluator = val_evaluator

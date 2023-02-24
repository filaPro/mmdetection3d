_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['projects.TD3D.td3d'])

model = dict(
    type='TD3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='TD3DMinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
        return_stem=True,
        stem_stride=1),
    neck=dict(
        type='TD3DNeck',
        in_channels=(64, 64, 128, 256, 512),
        out_channels=128,
        seg_out_channels=32),
    bbox_head=dict(
        type='TD3DDetectionHead',
        in_channels=128,
        voxel_size=0.02,
        padding=0.08,
        pts_assign_threshold=18,
        pts_center_threshold=8),
    seg_head=dict(
        type='TD3DSegmentationHead',
        voxel_size=0.02),
    train_cfg=dict(num_proposals=2),
    test_cfg=dict(
        nms_pre=1200,
        iou_thr=0.4,
        det_score_thr=0.1,
        seg_score_thr=0.2))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[28, 32],
    gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=33, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

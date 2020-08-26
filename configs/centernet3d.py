_base_ = [
    './_base_/datasets/kitti-3d-car.py', './_base_/schedules/cyclic_40e.py',
    './_base_/default_runtime.py'
]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size=[0.05, 0.05, 0.1]
num_class=1
checkpoint_config = dict(interval=5)
total_epochs = 25
evaluation = dict(interval=5)
lr = 0.000225
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.05)
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
model = dict(
    type='CenterNet3D',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SpMiddleFHD',
        in_channels=4,
        sparse_shape=[40, 1600, 1408],
        out_channels=128,
        activation="lrelu"),
    backbone=dict(
        type='RPN_SSD',
        in_channels=128,
        layer_nums=[3],
        layer_strides=[1],
        num_filters=[128],
        upsample_strides=[2],
        out_channels=[128],
        use_dcn=True,
        activation="lrelu"
    ),

    bbox_head=dict(
        type='Center3DHeadDepthAware',
        num_classes=num_class,
        in_channels=128,
        feat_channels=64,
        activation="lrelu",
        bbox_coder=dict(type='Center3DBoxCoder',num_class=num_class,
                        voxel_size=voxel_size,pc_range=point_cloud_range,
                        num_rad_bins=12,
                        downsample_ratio=4.0,
                        min_overlap=0.01,),
        loss_cls=dict(type='ModifiedFocalLoss',loss_weight=0.5),
        loss_xy=dict(type='GatherBalancedL1Loss',loss_weight=1.0),
        loss_z=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_dim=dict(type='GatherBalancedL1Loss', loss_weight=2.0),
        loss_dir=dict(type='GatherBinResLoss', loss_weight=1.0),
        loss_corner=dict(type='ModifiedFocalLoss', loss_weight=0.5),
        # loss_decode=dict("Boxes3dDecodeLoss",loss_weight=0.5),
        )
)
# model training and testing settings
train_cfg = dict()


test_cfg = dict(
    score_thr=0.20)

_base_ = ['./base_rtmdet_l.py',
          './base_settings_dior_rtmdet.py']
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
)
seed = 2024
"""
def 

"""
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

runner_type = 'MetaRemoveRunner'

train_cfg = dict(type='EpochBasedTrainLoop', val_interval=4)
frozen_parameters = ['backbone', ]

custom_imports = dict(
    imports=[
        'M_AD.engine.runner.meta_remove_runer',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
        'M_AD.datasets.samplers.one_task_sampler',

        'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
        'M_AD.models.detectors.E_Rtmdet_v2',
        'M_AD.models.dense_heads.E_Rrtmdet_head_v2',

        'M_AD.models.roi_heads.CLIP_VP_head_v1',
        'M_AD.models.necks.promopt_cspnext_pafpn',
    ],
    allow_failed_imports=False)

# ----- 视觉embed维度、分类embed维度、对齐维度（用来进行实例级别的对齐）
img_scale = (896, 896)
embed_dims = 256
# ----- 批次大小、各数据集采样率、使用gpu个数，每个epoch的最大迭代次数
batch_size = 4
source_prob = [16, 8, 8,
               8, 2, 2,
               2, 1, 4]
num_gpus = 4
max_iter_per_epoch = 10000
"""
数据集有：
D0_MAID,
D1_DOTA2,
D2_DIOR_R,
D3_FAIR1M,
D4_HRRSD,
D5_SpaceNet,
D6_Xview,
D7_HRSC2016,
D8_GLH_Bridge
"""

model = dict(
    with_aux_bbox_head=True,
    embed_dims=embed_dims,
    ###################
    type='OpenRTMDet',
    # -------- support 特征的信息
    ###############################
    pca_meta_pth='./data/7_25_pca_meta_DINOv2_256.pkl',
    support_feat_dict={
        'D1_DOTA2': './data/DOTA2_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D2_DIOR_R': './data/DIOR_R_dota/train_val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D3_FAIR1M': './data/FAIR1M_1024_0/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D4_HRRSD': './data/HRRSD_800_0/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D5_SpaceNet': './data/Spacenet_Merge/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D6_Xview': './data/xView_800_600/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D7_HRSC2016': './data/HRSC2016_DOTA/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'D8_GLH_Bridge': './data/GLH-Bridge_1024_200/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    },
    val_support_classes=['airplane', 'airport', 'baseballfield', 'basketballcourt',
                         'bridge', 'chimney', 'dam',
                         'Expressway-Service-area', 'Expressway-toll-station',
                         'golffield', 'groundtrackfield',
                         'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                         'tenniscourt', 'trainstation', 'vehicle', 'windmill'],
    val_dataset_flag='D2_DIOR_R',
    ################
    with_image_rec_losses=True,
    ################
    # -------- head
    bbox_head=dict(
        type='OpenRotatedRTMDetSepBNHead',
        embed_dims=embed_dims,
        num_classes=21,
        #############
        with_obj_align=False,
        with_slot_embed=True,
        cross_mlp_dim=2048,
        cross_num_layers=3,
    ),
    train_cfg=dict(
        assigner=dict(
            type='SafeDynamicSoftLabelAssigner')
    ),
)

############################### ----------- Dataset and Data_Loaders

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxTypeSafe', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    ########### ------ RandomRotate会导致empty instance的出现
    dict(
        type='RandomRotate',
        prob=0.25,
        angle_range=180),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputsMM')
]

train_class_name = ['SkyScript', ]
train_metainfo = dict(
    classes=train_class_name,
    # 注意：这个字段在最新版本中换成了小写
    palette=[(220, 20, 60),]
    # 画图时候的颜色，随便设置即可
)

D0_MAID = dict(
    type='DOTADatasetOnline',
    data_root='data/million_aid',
    dataset_flag='D0_MAID',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='test_png/'),
    ##########################
    ann_file='Step8_Remain_HighResolutions',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

D1_DOTA2 = dict(
    type='DOTADatasetOnline',
    dataset_flag='D1_DOTA2',
    data_root='data/DOTA2_800_600/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

D2_DIOR_R = dict(
    type='DOTADatasetOnline',
    dataset_flag='D2_DIOR_R',
    data_root='data/DIOR_R_dota/train_val',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

D3_FAIR1M = dict(
    type='DOTADatasetOnline',
    dataset_flag='D3_FAIR1M',
    data_root='data/FAIR1M_1024_0/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
D4_HRRSD = dict(
    type='DOTADatasetOnline',
    dataset_flag='D4_HRRSD',
    data_root='data/HRRSD_800_0/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
D5_SpaceNet = dict(
    type='DOTADatasetOnline',
    dataset_flag='D5_SpaceNet',
    data_root='data/Spacenet_Merge',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
D6_Xview = dict(
    type='DOTADatasetOnline',
    dataset_flag='D6_Xview',
    data_root='data/xView_800_600',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
D7_HRSC2016 = dict(
    type='DOTADatasetOnline',
    dataset_flag='D7_HRSC2016',
    data_root='data/HRSC2016_DOTA/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
D8_GLH_Bridge = dict(
    type='DOTADatasetOnline',
    dataset_flag='D8_GLH_Bridge',
    data_root='data/GLH-Bridge_1024_200/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file='Step6_Format_labels',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)


train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='OneTaskSampler',
        batch_size=batch_size,
        source_prob=source_prob,
        num_gpus=num_gpus,
        max_iter_per_epoch=max_iter_per_epoch
    ),
    dataset=dict(
        type='mmdet.ConcatDataset', datasets=[D0_MAID,
                                              D1_DOTA2,
                                              D2_DIOR_R,
                                              D3_FAIR1M,
                                              D4_HRRSD,
                                              D5_SpaceNet,
                                              D6_Xview,
                                              D7_HRSC2016,
                                              D8_GLH_Bridge]))






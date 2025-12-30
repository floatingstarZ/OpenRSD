_base_ = ['./base_rtmdet_l.py',
          './base_settings_dior_rtmdet.py']

#########################
val_img_scale = (800, 800)
val_support_classes = ['airport', 'baseball-diamond', 'basketball-court',
                       'bridge', 'container-crane', 'ground-track-field',
                       'harbor', 'helicopter', 'helipad', 'large-vehicle',
                       'plane', 'roundabout', 'ship', 'small-vehicle',
                       'soccer-ball-field', 'storage-tank', 'swimming-pool', 'tennis-court']
val_dataset_flag = 'Data1_DOTA2'

metainfo = dict(
    classes=val_support_classes,
    palette=[(220, 20, 60), ]
)

file_client_args = dict(backend='disk')
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=val_img_scale, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=val_img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=val_img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=val_img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root='./data/DOTA2_1024_500',
        metainfo=metainfo,
        ann_file='ss_val/annfiles',
        data_prefix=dict(img_path='ss_val/images'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader
#########################

load_from = './results/MMR_AD_A08_e_rtm_v2_base_recheck/epoch_36.pth'
frozen_parameters = ['backbone', 'neck']# ['backbone.stem']
batch_size = 2
data_root = '/data/space2/huangziyue'

# load_from = './results/MMR_AD_A08_e_rtm_v2_base_recheck/epoch_36.pth'
# frozen_parameters = ['backbone.stem']
# batch_size = 4
# data_root = '/gpfsdata/home/huangziyue/data'


max_epochs = 24
base_lr = 0.004 / max_epochs

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
)
seed = 2024
"""
继承A09_flex_rtm_v1_base
1. 加入两个新数据集
2. 增加Iteration到12000
3. 不使用background 
4. 更快的负样本极端

"""

runner_type = 'MetaRemoveRunner'

custom_imports = dict(
    imports=[
        'M_AD.engine.runner.meta_remove_runer',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
        'M_AD.datasets.samplers.one_task_sampler',

        'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
        'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal',
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1',

        'M_AD.models.roi_heads.CLIP_VP_head_v1',
        'M_AD.models.necks.promopt_cspnext_pafpn',
        'M_AD.evaluation.metrics.detail_dota_metric',

    ],
    allow_failed_imports=False)

# ----- 输出更详细的指标
val_evaluator = dict(type='DETAILDOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# ----- 视觉embed维度、分类embed维度、对齐维度（用来进行实例级别的对齐）
img_scale = (832, 832)
embed_dims = 256
# --- labelled, pure image
source_prob = [8, 2, 8, 2,
               2, 0.5, 1, 4,
               2, 0.5]
num_gpus = 4
max_iter_per_epoch = 12000
"""
# D0_MAID,
D1_DOTA2,
D2_DIOR_R,
D3_FAIR1M,
# D4_HRRSD,
D5_SpaceNet,
D6_Xview,
D7_HRSC2016,
D8_GLH_Bridge
D9_FMoW
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
        'Data1_DOTA2': './data/DOTA2_1024_500/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data1_DOTA1': './data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data2_DIOR_R': './data/DIOR_R_dota/train_val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data3_FAIR1M': './data/FAIR1M_2_800_400/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data4_HRRSD': './data/TGRS_HRRSD/train_val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data5_SpaceNet': './data/Spacenet_Merge/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data6_Xview': './data/xView_New_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data7_HRSC2016': './data/HRSC2016_DOTA/train/Step5_3_Prepare_Visual_Text_DINOv2_support_New.pkl',
        'Data8_GLH_Bridge': './data/GLH-Bridge_1024_200/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data9_FMoW': './data/FMoW/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data11_WHU_Mix': './data/WHU_Mix/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data12_ShipImageNet': './data//ShipRSImageNet_DOTA/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
        'Data28_STAR': './data/STAR_800_200/val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    },
    val_support_classes=val_support_classes,
    val_dataset_flag=val_dataset_flag,
    neg_support_data='./data/Neg_supports_v2.pkl',
    normalized_class_dict='./data/normalized_class_dict.pkl',
    max_neg_sample=36,
    val_using_aux=False,
    support_type='random',
    ################
    with_image_rec_losses=True,
    ################
    # -------- head
    bbox_head=dict(
        type='OpenRotatedRTMDetSepBNHead',
        embed_dims=embed_dims,
        num_classes=21,
        #############
        with_obj_align=True,
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
        prob=0.5,
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

Data1_DOTA2 = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data1_DOTA2',
    data_root='data/DOTA2_1024_500/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data1_DOTA2',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

Data2_DIOR_R = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data2_DIOR_R',
    data_root='data/DIOR_R_dota/train_val',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data2_DIOR_R',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

Data3_FAIR1M = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data3_FAIR1M',
    data_root='data/FAIR1M_2_800_400/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data3_FAIR1M',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
Data5_SpaceNet = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data5_SpaceNet',
    data_root='data/Spacenet_Merge/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data5_SpaceNet',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
Data6_Xview = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data6_Xview',
    data_root='data/xView_New_800_600/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data6_Xview',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
Data7_HRSC2016 = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data7_HRSC2016',
    data_root='data/HRSC2016_DOTA/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data7_HRSC2016',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)
Data8_GLH_Bridge = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data8_GLH_Bridge',
    data_root='data/GLH-Bridge_1024_200/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data8_GLH_Bridge',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

Data9_FMoW = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data9_FMoW',
    data_root='data/FMoW/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data9_FMoW',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

Data11_WHU_Mix = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data11_WHU_Mix',
    data_root='data/WHU_Mix/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data11_WHU_Mix',
    embed_dims=embed_dims,
    ##########################
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline
)

Data12_ShipImageNet = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data12_ShipImageNet',
    data_root='data/ShipRSImageNet_DOTA/train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ##########################
    ann_file=f'{data_root}/Formatted_FederatedLabels/Data12_ShipImageNet',
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
        type='mmdet.ConcatDataset', datasets=[Data1_DOTA2,
                                              Data2_DIOR_R,
                                              Data3_FAIR1M,

                                              Data5_SpaceNet,
                                              Data6_Xview,
                                              Data7_HRSC2016,

                                              Data8_GLH_Bridge,
                                              Data9_FMoW,
                                              Data11_WHU_Mix,

                                              Data12_ShipImageNet
                                              ]))






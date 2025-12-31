_base_ = ['./base_orcnn.py',
          './base_settings_dior_two_stage.py']


train_class_name = ['airplane', 'airport', 'baseballfield', 'basketballcourt',
                    'bridge', 'chimney', 'dam',
                    'Expressway-Service-area', 'Expressway-toll-station',
                    'golffield', 'groundtrackfield',
                    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                    'tenniscourt', 'trainstation', 'vehicle', 'windmill',
                    'SAM_Obj']
metainfo = dict(
    classes=train_class_name,
    # 注意：这个字段在最新版本中换成了小写
    palette=[(220, 20, 60),]
    # 画图时候的颜色，随便设置即可
)

# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/DIOR_R_dota/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.0,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_val/GT_with_SAM_labelTxt_5_23_IOU_01/',
        data_prefix=dict(img_path='train_val/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

max_epochs = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=65534)

custom_imports = dict(
    imports=[
        'M_AD.models.detectors.Tool_feat_patch_extractor_DINOv2',
    ],
    allow_failed_imports=False)


model = dict(
    type='ToolFeatExtractor',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False),
    out_dir='/data/space2/huangziyue/DIOR_R_dota/train_val/7_18_Extract_Feats_DIOR_SAM_with_GT_DINOv2_ViTL',
    bbox_roi_extractor=dict(
        out_size=(224, 224),
        spatial_scale=1.0,
        sampling_ratio=2,
        clockwise=True
    ),
    roi_scale_factor=1.25,
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(train_class_name),))
)



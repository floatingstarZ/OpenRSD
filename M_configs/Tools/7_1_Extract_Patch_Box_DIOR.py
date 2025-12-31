_base_ = ['./base_orcnn.py',
          './base_settings_dior_two_stage.py']

max_epochs = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=65534)

custom_imports = dict(
    imports=[
        'M_AD.models.detectors.Tool_img_patch_with_box_extractor',
    ],
    allow_failed_imports=False)

model = dict(
    type='ToolImgPatchWithBoxExtractor',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False),
    out_dir='/opt/data/nfs/huangziyue/DIOR_R_dota/train_val/7_1_Extract_Patch_Box_DIOR',
    bbox_roi_extractor=dict(
        out_size=(224, 224),
        spatial_scale=1.0,
        sampling_ratio=2,
        clockwise=True
    ),
    roi_scale_factor=1.25,
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,))
)
train_dataloader = dict(
    batch_size=1,
    num_workers=1)




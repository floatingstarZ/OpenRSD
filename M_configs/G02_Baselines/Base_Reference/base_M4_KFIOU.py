_base_ = './base_M1_RtnNetOBB.py'

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        loss_bbox_type='kfiou',
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)),
    train_cfg=dict(
        assigner=dict(iou_calculator=dict(type='FakeRBboxOverlaps2D'))))

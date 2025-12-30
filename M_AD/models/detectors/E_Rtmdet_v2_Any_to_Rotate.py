# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torchvision.ops import boxes

from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.registry import MODELS

from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import List, Tuple, Union
from torch import Tensor

from mmengine.structures.instance_data import InstanceData
from copy import deepcopy
from ctlib.os import *
import torch.nn.functional as F
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmdet.structures.bbox import bbox2roi
from mmrotate.structures.bbox import rbbox_overlaps
from mmdet.models.detectors import RTMDet
import math
from mmengine.model import xavier_init, normal_init

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                                 DinoTransformerDecoder, SinePositionalEncoding)
from mmdet.models.detectors.deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from mmdet.models.detectors.dino import DINO
import mmrotate.registry as mmr_registry
import mmdet.registry as mmd_registry
from copy import deepcopy

from ctlib.rbox import obb2xyxy
import mmdet
from mmrotate.structures.bbox.rotated_boxes import RotatedBoxes
def check_angle(angle):
    period = math.pi
    angle = (angle + period / 2) % period - period / 2
    return angle

###############
from M_AD.models.layers.transformer.dinor_layersv2 import (CdnRQueryGenerator,
                                                           DinoRTransformerDecoder)
from mmrotate.structures import RotatedBoxes, distance2obb

def obb2hbb(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs : [x_ctr,y_ctr,w,h,angle]

    Returns:
        outer hbb in obb format
    """
    N = obboxes.shape[0]
    if N == 0:
        return obboxes
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    xyxy = torch.cat([center - bias, center + bias], dim=-1)
    x1, y1, x2, y2 = torch.split(xyxy, [1, 1, 1, 1], dim=-1)
    a = torch.zeros_like(x1).to(obboxes.device)
    hbb = torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, a], dim=-1)

    return hbb

def add_bbox_noises(gt_bboxes, img_shape, noise_scale):
    device = gt_bboxes.device
    H, W = img_shape
    scale_factor = torch.tensor([H, W, H, W]).to(device)
    gt_bboxes[:, :4] = gt_bboxes[:, :4] / scale_factor

    # xywh -> xyxy
    bboxes = torch.cat([gt_bboxes[:, 0:2] - gt_bboxes[:, 2:4] / 2,
                        gt_bboxes[:, 0:2] + gt_bboxes[:, 2:4] / 2], dim=-1)

    # bbox_noise: whwh / 2 * noise, noise in [-box_noise_scale, +box_noise_scale]
    diff = torch.cat([gt_bboxes[:, 2:4], gt_bboxes[:, 2:4]], dim=-1) / 2
    noise = torch.rand_like(bboxes).to(device) * 2 - 1
    bboxes = bboxes + noise * diff * noise_scale
    bboxes = bboxes.clamp(min=0.0, max=1.0)
    out_bboxes = torch.cat([(bboxes[:, :2] + bboxes[:, 2:]) / 2,
                            bboxes[:, 2:] - bboxes[:, :2]], dim=-1)

    # angle_noise:
    angle = gt_bboxes[:, 4:5]
    angle_noise = torch.rand_like(angle).to(device) * 2 - 1
    angle = angle + angle_noise * noise_scale
    out_angle = check_angle(angle)

    out_gt_bboxes = torch.cat([out_bboxes, out_angle], dim=-1)
    out_gt_bboxes[:, :4] = out_gt_bboxes[:, :4] * scale_factor

    return out_gt_bboxes


@mmr_registry.MODELS.register_module()
class AceRTMDet(RTMDet):
    """
    v8效果不好，考虑引入LoRA、Grafting来让两个网络区分性更大
    """

    def __init__(self,
                 *args,
                 roi_neck: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 roi_train_cfg: OptConfigType = None,
                 roi_test_cfg: OptConfigType = None,
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.roi_neck = mmd_registry.MODELS.build(roi_neck)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = roi_train_cfg.rcnn if roi_train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=roi_test_cfg.rcnn)
            self.roi_head = mmd_registry.MODELS.build(roi_head)

        self.roi_train_cfg = roi_train_cfg
        self.roi_test_cfg = roi_test_cfg

    def extract_feat(self, batch_inputs: Tensor):
        x = self.backbone(batch_inputs)
        feat_stage1 = x[0]
        x = list(x[1:])
        if self.with_neck:
            x = self.neck(x)
        return x, feat_stage1

    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # print(batch_inputs.shape)

        for samples in batch_data_samples:
            boxes = samples.gt_instances.bboxes
            labels = torch.zeros(len(boxes)).long().to(boxes.device)
            samples.gt_instances.labels = labels

        losses = dict()
        # ----- YOLO branch, with YOLO Backbone, Neck, and Head
        yolo_feats, feat_stage1 = self.extract_feat(batch_inputs)
        yolo_losses = self.bbox_head.loss(yolo_feats, batch_data_samples)
        predicts = self.bbox_head.predict(yolo_feats,
                                          batch_data_samples,
                                          rescale=False,
                                          nms_pre=2000,
                                          iou_threshold=0.1,
                                          max_per_img=1000,
                                          score_thr=0.0)
        losses.update(yolo_losses)
        # ----- RoI branch, with RoI Neck and Head
        roi_feats = [feat_stage1, *yolo_feats]
        roi_feats = self.roi_neck(roi_feats)

        # ----- trans predicted rboxes to hboxes
        new_predicts = []
        for i, instance_data in enumerate(predicts):
            # --- predicted boxes -> hboxes
            rboxes = instance_data.bboxes.tensor.detach()
            hboxes = obb2hbb(rboxes)
            # --- gt -> hbox gts
            gt_rboxes = batch_data_samples[i].gt_instances.bboxes.tensor.detach()
            gt_hboxes = obb2hbb(gt_rboxes)
            gt_scores = torch.ones(len(gt_hboxes)).float().to(rboxes.device)
            gt_labels = torch.zeros(len(gt_hboxes)).long().to(rboxes.device)

            bboxes = torch.cat([gt_hboxes, hboxes])
            # print(bboxes)
            scores = torch.cat([gt_scores, instance_data.scores])
            labels = torch.cat([gt_labels, instance_data.labels])

            results = InstanceData()
            results.bboxes = RotatedBoxes(bboxes)
            results.scores = scores
            results.labels = labels
            new_predicts.append(results)

        roi_losses = self.roi_head.loss(roi_feats, new_predicts, batch_data_samples)

        for k, v in roi_losses.items():
            losses[f'RoI_{k}'] = v

        return losses

    # def predict(self,
    #             batch_inputs: Tensor,
    #             batch_data_samples: SampleList,
    #             rescale: bool = True) -> SampleList:
    #     # ----- YOLO branch, with YOLO Backbone, Neck, and Head
    #     yolo_feats, feat_stage1 = self.extract_feat(batch_inputs)
    #     predicts = self.bbox_head.predict(yolo_feats,
    #                                       batch_data_samples,
    #                                       rescale=False,
    #                                       nms_pre=2000,
    #                                       iou_threshold=None,
    #                                       max_per_img=1000,
    #                                       score_thr=0.0)
    #     # ----- RoI branch, with RoI Neck and Head
    #     roi_feats = [feat_stage1, *yolo_feats]
    #     roi_feats = self.roi_neck(roi_feats)
    #     results_list = self.roi_head.predict(roi_feats, predicts, batch_data_samples,
    #                                          rescale=rescale)
    #
    #     batch_data_samples = self.add_pred_to_datasample(
    #         batch_data_samples, results_list)
    #     return batch_data_samples

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples,
                rescale=True,
                gt_obb2hbb=False,
                sample_time=33,
                noise_scale=0.2) -> SampleList:
        # ----- YOLO branch, with YOLO Backbone, Neck, and Head
        yolo_feats, feat_stage1 = self.extract_feat(batch_inputs)
        device = yolo_feats[0].device
        roi_feats = [feat_stage1, *yolo_feats]
        roi_feats = self.roi_neck(roi_feats)

        all_results_list = []
        out_proposals_list = []

        for j in range(sample_time):
            proposal_results = []
            for i, sample in enumerate(batch_data_samples):
                bboxes = deepcopy(sample.gt_instances.bboxes.tensor)
                if gt_obb2hbb:
                    bboxes = obb2hbb(bboxes)
                noise_bboxes = add_bbox_noises(bboxes, (2048, 2048), 
                                               noise_scale=noise_scale)
                scores = torch.ones(len(noise_bboxes)).float().to(device)
                labels = torch.zeros(len(noise_bboxes)).long().to(device)

                results = InstanceData()
                results.bboxes = noise_bboxes
                results.scores = scores
                results.labels = labels
                proposal_results.append(results)
            #-----不适用NMS操作
            results_list = self.roi_head.predict(
                roi_feats, proposal_results, batch_data_samples, rescale=rescale)
            all_results_list.append(results_list)
            out_proposals_list.append(proposal_results)

        return all_results_list, out_proposals_list




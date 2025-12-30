# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
import mmrotate.registry as mmr_registry
from mmengine.model import BaseModule
from torch.nn.modules.utils import _pair
from mmdet.models.roi_heads import BaseRoIHead
from mmdet.structures.bbox import bbox2roi
import torch.nn.functional as F
from mmengine.model import bias_init_with_prob, constant_init, normal_init
import torch
from copy import deepcopy
import math
from ctlib.rbox import obb2xyxy
import mmdet
from mmrotate.structures.bbox.rotated_boxes import RotatedBoxes
def check_angle(angle):
    period = math.pi
    angle = (angle + period / 2) % period - period / 2
    return angle

def add_bbox_noises(gt_bboxes, img_shape):
    device = gt_bboxes.device
    H, W = img_shape
    scale_factor = torch.tensor([H, W, H, W]).to(device)
    gt_bboxes[:, :4] = gt_bboxes[:, :4] / scale_factor

    # random to hbox
    xyxy = obb2xyxy(gt_bboxes)
    x1, y1, x2, y2 = torch.split(xyxy, [1, 1, 1, 1], dim=-1)
    a = torch.zeros_like(x1).to(device)
    hbb = torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, a], dim=-1)
    rand_hbb_idx = torch.rand(len(xyxy)).to(device) > 0.7
    gt_bboxes[rand_hbb_idx] = hbb[rand_hbb_idx]

    # xywh -> xyxy
    bboxes = torch.cat([gt_bboxes[:, 0:2] - gt_bboxes[:, 2:4] / 2,
                        gt_bboxes[:, 0:2] + gt_bboxes[:, 2:4] / 2], dim=-1)

    # bbox_noise: whwh / 2 * noise, noise in [-box_noise_scale, +box_noise_scale]
    diff = torch.cat([gt_bboxes[:, 2:4], gt_bboxes[:, 2:4]], dim=-1) / 2
    noise = torch.rand_like(bboxes).to(device) * 2 - 1
    bboxes = bboxes + noise * diff * 0.3
    bboxes = bboxes.clamp(min=0.0, max=1.0)
    out_bboxes = torch.cat([(bboxes[:, :2] + bboxes[:, 2:]) / 2,
                            bboxes[:, 2:] - bboxes[:, :2]], dim=-1)

    # angle_noise:
    angle = gt_bboxes[:, 4:5]
    angle_noise = torch.rand_like(angle).to(device) * 2 - 1
    angle = angle + angle_noise * 0.2
    out_angle = check_angle(angle)

    out_gt_bboxes = torch.cat([out_bboxes, out_angle], dim=-1)
    out_gt_bboxes[:, :4] = out_gt_bboxes[:, :4] * scale_factor

    return out_gt_bboxes

@mmr_registry.MODELS.register_module()
class VisualPromptHead(BaseModule):
    """
    相比v3：
    引入框类型转换模块：
    1. 任意的Proposal（R-box+扰动）回归到R-box的分支
    2. 任意的Proposal（R-box+扰动）回归到H-box的分支
    """
    def __init__(self,
                 bbox_roi_extractor,
                 in_channels=768,
                 embed_dims=768,
                 backward_grad=0.05,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.featmap_strides = bbox_roi_extractor['featmap_strides']

        self.attn_convs = nn.ModuleList()
        self.any_to_rbox_convs = nn.Sequential(
                ConvModule(
                    192, in_channels,
                    3,1,1,
                    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                    act_cfg=dict(type='SiLU')),
                ConvModule(
                    in_channels, in_channels,
                    3, 1, 1,
                    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                    act_cfg=dict(type='SiLU')),
            )

        for s in self.featmap_strides:
            attn_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                          1,
                          3,
                          1,
                          1)
            )
            self.attn_convs.append(attn_conv)

        self.fc_visual = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_channels, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )
        self.fc_any_to_rbox = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_channels * 7 * 7, embed_dims),
            nn.SELU(inplace=True),
            nn.Linear(embed_dims, 5)
        )
        self.backward_grad = backward_grad
        self.bbox_coder = mmdet.registry.TASK_UTILS.build(
            dict(type='DeltaXYWHTRBBoxCoder',
                 angle_version='le90',
                 norm_factor=None,
                 edge_swap=True,
                 proj_xy=True,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(0.2, 0.2, 0.4, 0.4, 0.2)))
        self.reg_loss = mmdet.registry.MODELS.build(
            dict(type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))

        # for conv in self.embed_convs:
        #     normal_init(conv, std=0.01, bias=0)

    def forward(self, x, gt_boxes, head):
        # --- 多尺度融合（Grad Control） -> roi特征提取 ->
        # FC + Normalization -> 获得object embeddings

        # ---- gt_boxes添加噪声（随机变成Hbox+扰动），获得proposals
        proposals = [add_bbox_noises(deepcopy(b.tensor.detach()),(1024, 1024))
                     for b in gt_boxes]
        rois = bbox2roi(proposals)

        visual_feats = []
        reg_feats = []
        for i, f in enumerate(x):
            cls_feat = f #  * 0.1 + f.detach() * 0.9
            for cls_layer in head.cls_convs[i]:
                cls_feat = cls_layer(cls_feat)
            pred_embed = head.rtm_cls[i](cls_feat)

            attn_mask = self.attn_convs[i](pred_embed).sigmoid()
            v_feat = attn_mask * pred_embed
            visual_feats.append(v_feat)
            reg_feat = f # f * 0.3 + f.detach() * 0.7
            reg_feats.append(self.any_to_rbox_convs(reg_feat))

        # ----- 获得embeddings
        roi_feats = self.bbox_roi_extractor(
            visual_feats[:self.bbox_roi_extractor.num_inputs], rois)
        roi_feats = self.fc_visual(roi_feats)
        embeds = head.text_fc(roi_feats)
        embeds = F.normalize(embeds, dim=-1)

        # ----- 获得回归特征
        reg_roi_feats = self.bbox_roi_extractor(
            reg_feats[:self.bbox_roi_extractor.num_inputs], rois)
        reg_pred = self.fc_any_to_rbox(reg_roi_feats)

        # ---- 计算回归损失
        prior_boxes = RotatedBoxes(torch.cat(proposals))
        target_boxes = RotatedBoxes(torch.cat([deepcopy(b.tensor.detach()) for b in gt_boxes]))
        reg_target = self.bbox_coder.encode(prior_boxes, target_boxes)
        bbox_weights = torch.ones([len(reg_target), 5]).to(reg_target.device)
        reg_loss = self.reg_loss(reg_target,
                                 reg_pred,
                                 bbox_weights,
                                 reg_target.size(0))

        n_img = len(proposals)
        batch_ids = rois[:, 0]
        object_embeds = []
        for i in range(n_img):
            in_batch_ids = batch_ids == i
            object_embeds.append(embeds[in_batch_ids])
        return object_embeds, reg_loss

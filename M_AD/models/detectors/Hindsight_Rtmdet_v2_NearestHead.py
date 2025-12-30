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
from mmcv.ops.nms import nms_rotated
from ctlib.transform import to_array
from mmengine.structures.instance_data import InstanceData
from mmrotate.structures import RotatedBoxes, distance2obb
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
import mmdet.registry as mmd_registry
from mmrotate.structures.bbox import rbbox_overlaps
import math

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

# from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.layers import CSPLayer
from mmrotate.registry import MODELS
from mmdet.models.necks import CSPNeXtPAFPN
import math
from typing import Tuple, Type
from M_AD.models.utils.transformer_modular_yolo import TwoWayTransformerModularYOLO
from ctlib.os import *
from ctlib.transform import to_array
from M_Loggers.loggers import GloLogger
from M_Loggers.utils import AutoLogger
from copy import deepcopy
from M_AD.models.dense_heads.D_Rrtmdet_head_Simple_Base_v1 import OpenRotatedRTMDetSepBNHeadSimpleBase
from functools import reduce
from ctlib.rbox import obb2xyxy
from M_AD.models.roi_heads.Visual_Prompt_head_v7 import VisualPromptHead
from M_AD.models.utils.gather import gather_tensors, concat_all_gather, concat_all_gather_diff_size

"""
基于Hindsight_Rtmdet_v1_QueryHead
1. 改成和Gen_Rtmdet_v8_Ex_ICTHead_v6_RTVFast一样的KNN
2. 对Head的Alignment损失进行改进，改成类似的分类损失
3. Box Prompt 和 检测头 设置独立的Fusion Neck，让Retrieval和检测任务都能更好的进行，最后采用了独立的Head
4. 添加额外的检测
"""


class MultiLevelFusionDecoder(nn.Module):
    r"""
    Multi-level fused semantic segmentation head.
    from mmdet.roi_heads.mask_heads.fused_semantic_head.py import FusedSemanticHead
    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  # /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  # \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins=3,
                 fusion_level=0,
                 in_channels=192,
                 loss_rec=dict(
                     type='mmdet.L1Loss',
                     loss_weight=1.0),
                 grad_ratio=0.05,
                 ):
        super(MultiLevelFusionDecoder, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            )
        self.pred_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 3, 1, 1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, 3, 3, 2, 1),
        )
        self.loss_rec = mmd_registry.MODELS.build(loss_rec)
        self.grad_ratio = grad_ratio

    def forward(self, input_feats):
        feats = [f * self.grad_ratio + f.detach() * (1 - self.grad_ratio)
                 for f in input_feats]

        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x = x + self.lateral_convs[i](feat)
        pred_rec = self.pred_conv(x)
        return pred_rec

    def loss(self, feats, org_imgs):
        pred_rec = self.forward(feats)
        # print(pred_rec.shape, org_imgs.shape)
        _, _, H, W = pred_rec.shape
        rec_target = nn.functional.interpolate(org_imgs,
                                               (H, W),
                                               mode='bilinear').detach()
        loss = self.loss_rec(pred_rec, rec_target)
        return loss


class ICT_Neck(nn.Module):
    """
    1. Backbone和Neck的特征进行融合
    2.
    """

    def __init__(self,
                 backbone_channels=[256, 512, 1024],
                 neck_channels=[256, 256, 256],
                 out_channels=[256, 256, 256],
                 ):
        super(ICT_Neck, self).__init__()
        # --- Fusion Convs
        self.b_l_convs = nn.ModuleList()  # backbone lateral convs
        self.n_l_convs = nn.ModuleList()  # neck lateral convs
        for b, n, o in zip(backbone_channels, neck_channels, out_channels):
            self.b_l_convs.append(
                nn.Sequential(
                    nn.Conv2d(b, o, 1, 1, 0)
                )
            )
            self.n_l_convs.append(
                nn.Sequential(
                    nn.Conv2d(n, o, 1, 1, 0)
                )
            )

    def forward(self, x_backbone, x_neck):
        # --- feature backbone/neck lateral
        x_b_l = [self.b_l_convs[i](f) for i, f in enumerate(x_backbone)]
        x_n_l = [self.n_l_convs[i](f) for i, f in enumerate(x_neck)]

        x_out = [x_b + x_n for x_b, x_n in zip(x_b_l, x_n_l)]

        return x_out


###################################################################################################################
@MODELS.register_module()
class OpenRTMDet(SingleStageDetector):
    """
    在v5的基础上：
    1. 将负样本从 全部数据集内其他类别 改为 采样
    2. 加入处理Image-Text的分支
    3. 统一以小写处理text
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True,
                 ###############################
                 support_type='random',
                 support_feat_dict=None,
                 normalized_class_dict=None,
                 val_support_classes=[],
                 val_dataset_flag='DIOR_R',
                 neg_support_data=None,
                 max_neg_sample=36,
                 ################
                 with_image_rec_losses=False,
                 pca_meta_pth='',
                 with_aux_bbox_head=True,
                 embed_dims=256,
                 ################
                 val_using_aux=False,
                 with_random_neg=True,
                 ################
                 with_ict_head=False,
                 box_prompt_head=None,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        ########## ict_head采用Fusion Head的结构
        # self.ict_bkb = deepcopy(self.backbone)
        self.ict_head = MODELS.build(bbox_head)
        self.ict_nck = MODELS.build(neck)
        self.ict_fusion_module = ICT_Neck()
        from M_AD.models.necks.Ace_fpn import AceFPN

        self.box_nck = MODELS.build(neck)
        self.box_fusion_module = ICT_Neck()
        self.box_fpn = AceFPN(in_channels=[128, 256, 256, 256],
                              out_channels=256,
                              num_outs=4)
        self.box_prompt_head = MODELS.build(box_prompt_head)
        ####################
        # TODO: Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

        # ---- 图像重建损失
        self.with_image_rec_losses = with_image_rec_losses
        if self.with_image_rec_losses:
            self.rec_neck = MultiLevelFusionDecoder(
                in_channels=self.bbox_head.in_channels)

        # ---- 构造uni_support_data，对类别名称进行归一化
        self.norm_cls_map = pklload(normalized_class_dict)
        assert support_feat_dict is not None
        self.support_data_dict = dict()
        self.uni_support_data = dict()
        for dataset_flag, support_feat_pth in support_feat_dict.items():
            support_data = pklload(support_feat_pth)
            normed_support_data = {}
            for k, v in support_data.items():
                normed_k = self.norm_cls_map[k]
                if len(v['visual_embeds']) <= 10:
                    v['visual_embeds'] = np.concatenate([v['visual_embeds'] for i in range(10)])
                if len(v['text_embeds']) <= 10:
                    v['text_embeds'] = np.concatenate([v['text_embeds'] for i in range(10)])
                normed_support_data[normed_k] = v
                # ----- 构造统一的support
                if normed_k in self.uni_support_data.keys():
                    # ------ 合并同类名
                    self.uni_support_data[normed_k]['visual_embeds'] = np.concatenate([v['visual_embeds'],
                                                                                       self.uni_support_data[normed_k]
                                                                                       ['visual_embeds']])
                    self.uni_support_data[normed_k]['text_embeds'] = np.concatenate([v['text_embeds'],
                                                                                     self.uni_support_data[normed_k]
                                                                                     ['text_embeds']])
                    if get_world_size() > 1:
                        print(f'In uni_support_data: Merge Class {k} in {dataset_flag} into {normed_k}', 'current')
                        print_log(f'In uni_support_data: Merge Class {k} in {dataset_flag} into {normed_k}', 'current')
                else:
                    self.uni_support_data[normed_k] = v
            ### ---- 检查前后一致性
            if len(normed_support_data) != len(support_data):
                raise Exception(f'UniSupport Error | pre: {list(support_data.keys())}, '
                                f'Normed: {list(normed_support_data.keys())}')

            self.support_data_dict[dataset_flag] = deepcopy(normed_support_data)

        ############################ ---- uni_support_data按照顺序排列
        uni_names = sorted(list(self.uni_support_data.keys()))
        sorted_uni_support_data = OrderedDict()
        for name in uni_names:
            sorted_uni_support_data[name] = self.uni_support_data[name]
        self.uni_support_data = sorted_uni_support_data
        uni_classes = self.uni_support_data.keys()
        self.uni_name2id = {c: i for i, c in enumerate(uni_classes)}
        self.uni_id2name = {i: c for i, c in enumerate(uni_classes)}
        ############################ ---- 构造memory bank
        N_class = len(uni_names)
        queue_len = 256
        q_max_update = 64
        max_iter = 60000
        self.register_buffer('queue', torch.zeros(N_class, queue_len, 512))
        self.register_buffer('queue_label', torch.ones(N_class, queue_len).long() * -2)
        self.register_buffer('queue_time', torch.ones(N_class, queue_len).long() * -2)
        self.register_buffer('queue_ptr', torch.zeros(N_class, dtype=torch.long))
        self.q_max_update = q_max_update
        self.queue_len = queue_len
        self.max_iter = max_iter
        self.cur_iter = 0  # 迭代次数，代表了时间
        ############################ ---- 构造memory bank

        # ---- 获得预定义的cls -> Negative映射
        neg_support_data = pklload(neg_support_data)
        self.neg_mapping = neg_support_data['neg_dict']
        for dataset_flag, neg_maps in self.neg_mapping.items():
            neg_maps = {self.norm_cls_map[cls_name]: [self.norm_cls_map[c] for c in neg_list]
                        for cls_name, neg_list in neg_maps.items()}
            self.neg_mapping[dataset_flag] = neg_maps
        self.max_neg_sample = max_neg_sample
        ##############
        org_val_support_classes = deepcopy(val_support_classes)
        val_support_classes = [self.norm_cls_map[v] for v in val_support_classes]
        if len(set(val_support_classes)) != len(set(org_val_support_classes)):
            raise Exception(f'val_support_classes Error | pre: {set(org_val_support_classes)}, '
                            f'Normed: {set(val_support_classes)}')

        self.val_support_data = self.support_data_dict[val_dataset_flag]
        self.val_support_classes = val_support_classes
        self.val_name2id = {name: cat_id for cat_id, name in enumerate(val_support_classes)}
        self.val_id2name = {cat_id: name for cat_id, name in enumerate(val_support_classes)}

        ##############
        # self.visual_prompt_head = MODELS.build(visual_prompt_head)
        self.pca_meta = pklload(pca_meta_pth)
        ############################################        DINO -> embeds
        self.visual_support_mapping = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embed_dims)
        )
        self.text_support_mapping = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embed_dims)
        )
        self.box_support_mapping = nn.Sequential(
            nn.Linear(512, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )
        self.box_visual_to_semantic = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )
        self.support_type = support_type
        ########################################
        conv_cfg = None
        norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg = dict(type='Swish')
        if with_aux_bbox_head:
            self.aux_bbox_head = OpenRotatedRTMDetSepBNHeadSimpleBase(
                num_classes=20,
                in_channels=self.bbox_head.in_channels,
                stacked_convs=2,
                feat_channels=self.bbox_head.feat_channels,
                angle_version='le90',
                anchor_generator=dict(
                    type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
                bbox_coder=dict(
                    type='DistanceAnglePointCoder', angle_version='le90'),
                loss_cls=dict(
                    type='mmdet.QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0),
                loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
                with_objectness=False,
                exp_on_reg=True,
                share_conv=True,
                pred_kernel_size=1,
                use_hbbox_loss=False,
                scale_angle=False,
                loss_angle=None,
                norm_cfg=dict(type='SyncBN'),
                act_cfg=dict(type='SiLU'),
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                ##################
                embed_dims=embed_dims,
                with_obj_align=self.bbox_head.with_obj_align,
            )
            self.aux_convs = nn.ModuleList()
            aux_channels = [self.bbox_head.feat_channels, ] * 3
            for channels in aux_channels:
                aux_conv = ConvModule(
                    channels,
                    channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.aux_convs.append(aux_conv)
        else:
            self.aux_bbox_head = None
        self.val_using_aux = val_using_aux
        self.count = 1
        self.with_random_neg = with_random_neg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        #### ---- 如果ict_head没在预训练权重中，则复制bbox_head的权重给它
        if not any([
            'ict_head' in key
            for key in state_dict.keys()
        ]):
            ict_head_state_dict = OrderedDict()
            for k, v in state_dict.items():
                first_name = k.split('.')[0]
                if first_name == 'bbox_head':
                    w_name = k[len('bbox_head.'):]
                    new_k = f'ict_head.{w_name}'
                    ict_head_state_dict[new_k] = v
                    # print(f'Copy {k} to {new_k}')
            print('Add New Keys: ', list(ict_head_state_dict.keys()))
            state_dict.update(ict_head_state_dict)
        #### ---- 如果ict_fusion_module没在预训练权重中，则复制neck的权重给它
        if not any([
            'ict_nck' in key
            for key in state_dict.keys()
        ]):
            ict_head_state_dict = OrderedDict()
            for k, v in state_dict.items():
                first_name = k.split('.')[0]
                if first_name == 'neck':
                    w_name = k[len('neck.'):]
                    new_k = f'ict_nck.{w_name}'
                    ict_head_state_dict[new_k] = v
                    # print(f'Copy {k} to {new_k}')
            print('Add New Keys: ', list(ict_head_state_dict.keys()))
            state_dict.update(ict_head_state_dict)

        def add_to_state_dict(old_key, new_key):
            if not any([
                f'{new_key}.' in key
                for key in state_dict.keys()
            ]):
                ict_head_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    first_name = k.split('.')[0]
                    if first_name == old_key:
                        w_name = k[len(f'{old_key}.'):]
                        new_k = f'{new_key}.{w_name}'
                        ict_head_state_dict[new_k] = v
                        # print(f'Copy {k} to {new_k}')
                # print('Add New Keys: ', list(ict_head_state_dict.keys()))
                state_dict.update(ict_head_state_dict)
        add_to_state_dict('neck', 'box_nck')

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extract_feat(self,
                     batch_inputs: Tensor,
                     return_backbone=False):
        x = self.backbone(batch_inputs)
        feat_stage1 = x[0]
        x_base = list(x[1:])
        x_neck = self.neck(x_base)

        if return_backbone:
            return feat_stage1, x_base, x_neck
        else:
            return x_neck

    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        ##### ------------ 判断数据类型
        dataset_flags = [sample.metainfo['dataset_flag']
                         for sample in batch_data_samples]
        if len(set(dataset_flags)) != 1:
            raise Exception(f'Only one dataset is supported, but get: {dataset_flags}')
        dataset_flag = dataset_flags[0]
        ##### ------------- 提取特征
        losses = dict()
        feat_stage1, x_base, x_neck = self.extract_feat(batch_inputs, return_backbone=True)
        x_base = list(x_base)
        x = list(x_neck)
        device = x[0].device
        ##### ------------- 统一名称
        if dataset_flag not in ['Data0_MAID']:
            for sample in batch_data_samples:
                texts = sample.gt_instances.texts
                texts_ = [self.norm_cls_map[cls_name] for cls_name in texts]
                sample.gt_instances.texts = texts_
                sample.cls_list = list(set([self.norm_cls_map[c] for c in sample.cls_list]))

        ##### ------------ 对Neck特征添加重建损失
        if self.with_image_rec_losses:
            loss_rec = self.rec_neck.loss(x, batch_inputs)
            losses['ImgRec_loss'] = loss_rec

        A = sum([x.view(-1)[0] for x in self.visual_support_mapping.parameters()])
        B = sum([x.view(-1)[0] for x in self.text_support_mapping.parameters()])
        x[0] = x[0] + A * 0.0 + B * 0.0
        ##### ------------ 分类讨论，进行forward
        if dataset_flag == 'Data0_MAID':
            PI_losses = self.loss_pure_img(x, batch_data_samples)
            losses.update(PI_losses)
        else:
            ########### ------------ 构造support_data
            pos_cls_names = []
            all_cls_names = []
            for sample in batch_data_samples:
                pos_cls_names.extend(sample.gt_instances.texts)
                all_cls_names.extend(sample.cls_list)
            neg_cls_names = set(all_cls_names) - set(pos_cls_names)
            if len(neg_cls_names) > 0:
                # ---- 负样本采样
                if self.with_random_neg:
                    num_neg = int(np.random.randint(0, len(neg_cls_names) + 1))
                    num_neg = min(num_neg, self.max_neg_sample)
                else:
                    num_neg = self.max_neg_sample
                neg_cls_names = np.random.permutation(np.array(list(neg_cls_names)))[:num_neg].tolist()
                # ---- 获得所有support_data
                valid_cls_names = set(pos_cls_names).union(set(neg_cls_names))
                det_support_data = {k: v for k, v in self.uni_support_data.items() if k in valid_cls_names}
            else:
                det_support_data = {k: v for k, v in self.uni_support_data.items() if k in set(pos_cls_names)}
            name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
            id2name = {cat_id: name for cat_id, name in enumerate(det_support_data.keys())}
            ########### ------------ 随机采样text或者visual prompt
            if self.support_type in ['text', 'visual']:
                support_type = self.support_type
            elif self.support_type in ['random']:
                p = float(np.random.rand(1)[0])
                support_type = 'text' if p >= 0.5 else 'visual'
            else:
                raise Exception(f'Unrecognized support type {self.support_type}')

            IT_losses, predictions = self.loss_labelled(x,
                                                        batch_data_samples,
                                                        support_data=deepcopy(det_support_data),
                                                        support_name2id=name2id,
                                                        support_type=support_type,
                                                        dataset_flag=dataset_flag)

            losses.update(IT_losses)
            #############################################################################
            ############### 匹配Predictions和GT
            all_ict_boxes = []
            all_ict_labels = []

            max_label = max(list(name2id.values()))
            obj_id = max_label + 100

            for i, sample in enumerate(batch_data_samples):
                gt_instances = sample.gt_instances
                pred_instances = predictions[i]

                gt_boxes = gt_instances.bboxes.tensor
                gt_labels = gt_instances.labels
                pred_boxes = pred_instances.bboxes.tensor
                pred_labels = pred_instances.labels
                pred_scores = pred_instances.scores
                pred_boxes = pred_boxes[pred_scores >= 0.3]
                pred_labels = pred_labels[pred_scores >= 0.3]

                if len(gt_boxes) == 0:
                    ict_boxes = torch.zeros([0, 5]).float().to(device)
                    ict_labels = torch.zeros([0]).long().to(device)
                    all_ict_boxes.append(ict_boxes)
                    all_ict_labels.append(ict_labels)
                    continue

                if len(pred_boxes) == 0:
                    num_sample = int(np.random.randint(0, 8))
                    sample_indices = torch.randperm(len(gt_boxes))[:num_sample]
                    ict_boxes = gt_boxes[sample_indices]
                    ict_labels = gt_labels[sample_indices]
                    all_ict_boxes.append(ict_boxes)
                    all_ict_labels.append(ict_labels)
                    continue

                ious = rbbox_overlaps(gt_boxes, pred_boxes)
                gt_m_ious, _ = torch.max(ious, dim=1)
                dt_m_ious, _ = torch.max(ious, dim=0)
                # print(gt_m_ious)
                # print(dt_m_ious)

                # ---- 2. Det和GT中存在的类别并集作为接下来的Prompt类别集合
                label_set = torch.unique(torch.cat([gt_labels, pred_labels]))

                # ---- 3. 根据Box的IoU以及Label一致性匹配Det和GT，获得G_m, G_um和D_um。
                # ---- 4. 每个类别中，G_m中采样0-7个，G_um中采样0-7个，加上些许Box扰动，作为TP Box
                # ---- 5. 从𝐷_𝑢𝑚中采样0-7个，作为FP Box
                ict_boxes = []
                ict_labels = []

                for l in label_set:
                    # ---- GT中采样Box
                    if torch.sum(gt_labels == l) != 0:

                        gt_m = gt_boxes[(gt_m_ious >= 0.5) & (gt_labels == l)]
                        num_sample = int(np.random.randint(0, 8))
                        sample_indices = torch.randperm(len(gt_m))[:num_sample]
                        sampled_gt_m = gt_m[sample_indices]

                        gt_um = gt_boxes[(gt_m_ious <= 0.1) & (gt_labels == l)]
                        num_sample = int(np.random.randint(3, 8))
                        sample_indices = torch.randperm(len(gt_um))[:num_sample]
                        sampled_gt_um = gt_um[sample_indices]

                        tp_box = torch.cat([sampled_gt_m, sampled_gt_um])
                    else:
                        tp_box = torch.zeros([0, 5]).to(device)
                    tp_labels = torch.Tensor([l] * len(tp_box)).to(device).long()
                    # ---- DT中采样Box
                    if torch.sum(pred_labels == l) != 0:
                        fp_box = pred_boxes[(dt_m_ious <= 0.1) & (pred_labels == l)]
                        num_sample = int(np.random.randint(3, 8))
                        sample_indices = torch.randperm(len(fp_box))[:num_sample]
                        fp_box = fp_box[sample_indices]
                    else:
                        fp_box = torch.zeros([0, 5]).to(device)
                    # ---- fp_labels统一设置为-2，参与到cross-Attention中，
                    fp_labels = torch.arange(obj_id, obj_id + len(fp_box)).to(device).long() * 0 - 2
                    obj_id += len(fp_box)

                    ict_boxes.append(tp_box)
                    ict_boxes.append(fp_box)
                    ict_labels.append(tp_labels)
                    ict_labels.append(fp_labels)
                ict_boxes = torch.cat(ict_boxes)
                ict_labels = torch.cat(ict_labels)
                all_ict_boxes.append(ict_boxes)
                all_ict_labels.append(ict_labels)
            all_ict_texts = []
            for labels in all_ict_labels:
                texts = []
                for l in labels:
                    l = int(l)
                    if l in id2name.keys():
                        texts.append(id2name[l])
                    else:
                        texts.append(f'UnknownObj_{l}')
                all_ict_texts.append(texts)
            ############# ict head
            # Image -> BACKBONE -> feat_stage1, x_base (x_ict_base),
            # x_ict_base -> ICT_NCK -> x_ict_neck
            # x_ict_neck + x_ict_base -> ICT_FUSION_MODULE -> ict_x -> ICT_HEAD   -> detections
            # x_ict_neck + x_ict_base -> BOX_FUSION_MODULE -> box_x + feat_stage1 -> BOX_FPN -> BOX_PROMPT_HEAD -> embeddings

            x_ict_base = x_base
            x_ict_neck = self.ict_nck(x_ict_base)
            x_box_neck = self.box_nck(x_ict_base)

            ict_x = self.ict_fusion_module(x_ict_base, x_ict_neck)
            box_x = self.box_fusion_module(x_ict_base, x_box_neck)
            box_x = self.box_fpn([feat_stage1, *box_x])

            kwargs = dict()
            ########### ------------ 构造support_data
            # ------ 1. 首先，与之前一样，根据GT的类别采样support
            # ------    一些neg类也参与到训练中，增强分类能力
            # ------ 2. 提取TP和FP的特征
            box_embeds = self.box_prompt_head.extract_embed(box_x,
                                                            all_ict_boxes)
            # ------ 3. 将TP和FP用ict_embeddings映射成support
            all_box_embeds = []
            all_box_support_labels = []
            for embeds, labels in zip(box_embeds, all_ict_labels):
                labels_, indices = torch.sort(labels)  # 排序
                embeds_ = embeds[indices]

                all_box_embeds.append(embeds_)
                all_box_support_labels.append(labels_)

            ######################## 4. 基础support也必不可少，但是减少采样数量，使得检测器能更有效获取ict特征
            base_support_feats = []
            base_support_labels = []
            real_support_shots = []
            support_shot = int(np.random.randint(1, 3))

            for cls_name, support_info in det_support_data.items():
                cls_id = name2id[cls_name]
                feats = support_info['visual_embeds']
                labels = np.ones(len(feats)) * cls_id

                ids = np.arange(len(labels))
                sample_ids = np.random.permutation(ids)[:support_shot]
                real_support_shots.append(len(sample_ids))

                base_support_feats.append(feats[sample_ids])
                base_support_labels.append(labels[sample_ids])

            base_support_feats = torch.Tensor(np.concatenate(base_support_feats)).to(device)
            base_support_labels = torch.Tensor(np.concatenate(base_support_labels)).to(device).long()
            base_support_feats = self.visual_support_mapping(base_support_feats)

            ######################## 5. 拼接不同的特征并进行padding

            max_support_len = max([len(f) for f in all_box_embeds])
            ict_support_feats = []
            ict_support_labels = []
            for f, l in zip(all_box_embeds, all_box_support_labels):
                ict_box_f = self.box_support_mapping(f)

                if len(f) == max_support_len:
                    ict_support_feats.append(torch.cat([ict_box_f, base_support_feats]))
                    ict_support_labels.append(torch.cat([l, base_support_labels]))
                    continue
                # ----- 将instance labels置为0
                # l[l > max_label] = -1
                n_pad = max_support_len - len(ict_box_f)
                pad_feats = torch.zeros([n_pad, ict_box_f.shape[-1]]).float().to(device)
                pad_labels = torch.ones(n_pad).long().to(device) * -1
                ict_support_feats.append(torch.cat([ict_box_f, base_support_feats, pad_feats]))
                ict_support_labels.append(torch.cat([l, base_support_labels, pad_labels]))
            ict_support_feats = torch.stack(ict_support_feats)
            ict_support_labels = torch.stack(ict_support_labels)

            all_obj_embeds = [sample.gt_instances.visual_embeds.float() for sample in batch_data_samples]
            all_obj_labels = [sample.gt_instances.ins_labels for sample in batch_data_samples]  # 实例区分

            ######################## 6. 计算损失

            kwargs['num_classes'] = len(det_support_data)
            kwargs['align_style'] = 'labelled'
            kwargs['num_in_classes'] = kwargs['num_classes']
            kwargs['support_shot'] = support_shot

            ict_losses = self.ict_head.loss(ict_x,
                                            batch_data_samples,
                                            ict_support_feats,
                                            ict_support_labels,
                                            ict_support_labels,
                                            all_obj_embeds,
                                            all_obj_labels,
                                            **kwargs)
            for k, v in ict_losses.items():
                losses[f'ICT_{k}'] = v

            ################################### 基于Memory的Prompt优化 ###################################
            uni_name2id = self.uni_name2id
            uni_id2name = self.uni_id2name

            ### ----- 获得之前存储的memory特征，所有类别都拼接到一起
            global_memory_feats = []
            global_memory_labels = []
            classes = list(name2id.keys())
            for cls_name in classes:
                cls_id = uni_name2id[cls_name]

                cls_feats = self.queue[cls_id]
                cls_labels = self.queue_label[cls_id]

                mem_feats = cls_feats[cls_labels >= 0]
                mem_labels = torch.ones(len(mem_feats)).long() * name2id[cls_name]

                global_memory_feats.append(mem_feats)
                global_memory_labels.append(mem_labels)
            global_memory_feats = torch.cat(global_memory_feats).to(device)
            global_memory_labels = torch.cat(global_memory_labels).to(device)

            ### ----- 合并batch内所有的ict特征与label，让这些特征能参与到优化当中
            local_memory_feats = torch.cat(all_box_embeds)
            local_memory_labels = torch.cat(all_box_support_labels)
            local_memory_batch_ids = torch.cat([torch.ones(len(feats)).long().to(device) * i
                                                for i, feats in enumerate(all_box_embeds)])

            pos_ids = local_memory_labels >= 0
            local_memory_feats = local_memory_feats[pos_ids]
            local_memory_batch_ids = local_memory_batch_ids[pos_ids]
            local_memory_labels = local_memory_labels[pos_ids]

            ### ----- 去掉当前图像特征，让网络必须根据其他图像的特征来进行预测
            rtv_in_batch_embeds = []
            rtv_in_batch_labels = []
            for b_id in range(len(all_box_embeds)):
                non_self_ids = local_memory_batch_ids != b_id
                non_self_feats = local_memory_feats[non_self_ids]
                non_self_labels = local_memory_labels[non_self_ids]

                rtv_feats = torch.cat([non_self_feats, global_memory_feats])
                rtv_labels = torch.cat([non_self_labels, global_memory_labels])

                rtv_in_batch_embeds.append(rtv_feats)
                rtv_in_batch_labels.append(rtv_labels)

            ### ----- 对每张图片的检测结果进行采样，作为后续query的依据
            all_query_boxes = []
            all_query_labels = []
            all_query_target_labels = []
            for i, sample in enumerate(batch_data_samples):
                gt_instances = sample.gt_instances
                gt_boxes = gt_instances.bboxes.tensor
                gt_labels = gt_instances.labels
                pred_instances = predictions[i]

                pred_boxes = pred_instances.bboxes.tensor
                pred_labels = pred_instances.labels
                pred_scores = pred_instances.scores
                pred_boxes = pred_boxes[pred_scores >= 0.3]
                pred_labels = pred_labels[pred_scores >= 0.3]

                # --- 采样一些box，每个类别随机采样最多7个，用做后续的Query
                if len(pred_boxes) == 0:
                    sampled_boxes = gt_boxes
                    sampled_labels = gt_instances.labels
                else:
                    sampled_boxes = []
                    sampled_labels = []
                    label_set = torch.unique(pred_labels)
                    for l in label_set:
                        in_cls_boxes = pred_boxes[pred_labels == l]
                        sample_indices = torch.randperm(len(in_cls_boxes))[:7]
                        in_cls_boxes = in_cls_boxes[sample_indices]
                        sampled_boxes.append(in_cls_boxes)
                        sampled_labels.append(pred_labels[pred_labels == l][sample_indices])
                    sampled_boxes = torch.cat(sampled_boxes)
                    sampled_labels = torch.cat(sampled_labels)
                # --- 如果sampled_boxes不为空，且GT不为空，则进行匹配
                # --- 如果max iou>=0.5，则视为匹配上，否则为-2
                if len(sampled_boxes) != 0 and len(gt_boxes) != 0:
                    ious = rbbox_overlaps(sampled_boxes, gt_boxes)
                    m_gt_ious, m_gt_ids = torch.max(ious, dim=-1)
                    target_labels_ = gt_labels[m_gt_ids]
                    target_labels_[m_gt_ious < 0.5] = -1
                else:
                    target_labels_ = torch.ones(len(sampled_boxes)).long().to(device) * -2

                all_query_boxes.append(sampled_boxes)
                all_query_labels.append(sampled_labels)
                all_query_target_labels.append(target_labels_)
            all_query_embeds = self.box_prompt_head.extract_embed(box_x,
                                                                  all_query_boxes)
            rtv_mem_feats = []
            rtv_mem_labels = []
            nn_losses = []

            for i, q_embeds in enumerate(all_query_embeds):
                q_labels = all_query_labels[i]  # 当前图像的query
                q_target_labels = all_query_target_labels[i]
                q_sort_idx = torch.argsort(q_labels)
                q_embeds = q_embeds[q_sort_idx]
                q_labels = q_labels[q_sort_idx]
                q_target_labels = q_target_labels[q_sort_idx]

                k_embeds = rtv_in_batch_embeds[i]  # 其他图片GT获得的RoI特征 + Memory特征
                if len(k_embeds) == 0:
                    rtv_mem_feats.append(torch.zeros([0, 512]).float().to(device))
                    rtv_mem_labels.append(torch.zeros([0]).long().to(device))
                    continue
                k_labels = rtv_in_batch_labels[i]
                k_sort_idx = torch.argsort(k_labels)
                k_embeds = k_embeds[k_sort_idx]
                k_labels = k_labels[k_sort_idx]

                q = self.box_prompt_head.embed_to_semantic(q_embeds)
                k = self.box_prompt_head.embed_to_semantic(k_embeds)
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)
                sims = q @ k.transpose(-1, -2)
                label_eq = torch.eq(q_labels.reshape(-1, 1),
                                    k_labels.reshape(1, -1)).int()

                ########################################################
                # # --- Pred类别内最相似的1个
                # in_topk_sims, in_topk_indices = (sims * label_eq).topk(1, dim=1, largest=True)
                # in_topk_indices = torch.unique(in_topk_indices)
                #
                # in_mem_labels = k_labels[in_topk_indices]
                # in_mem_feats = k_embeds[in_topk_indices]
                #
                # # --- Pred类别外最相似的1个
                # out_topk_sims, out_topk_indices = (sims * (1 - label_eq)).topk(1, dim=1, largest=True)
                # out_topk_indices = torch.unique(out_topk_indices)
                #
                # out_mem_labels = k_labels[out_topk_indices]
                # out_mem_feats = k_embeds[out_topk_indices]
                #
                # mem_feats = torch.cat([in_mem_feats, out_mem_feats])
                # mem_labels = torch.cat([in_mem_labels, out_mem_labels])

                #########################################################
                top_k = min(len(k_embeds), 3)
                # --- 计算Sim(F_DT, Memory)，Sim为余弦相似度，获得每个F_DT对应3个最相似的Memory的index。
                # --- Memory的index取并集，每个memory找与其最近的F_DT，使用VQVAE技巧让F_DT也可以获得梯度。并获得对应的label。
                topk_sims, topk_indices = sims.topk(top_k, dim=1, largest=True)
                topk_indices = torch.unique(topk_indices)

                mem_labels = k_labels[topk_indices]
                mem_feats = k_embeds[topk_indices]

                rtv_mem_feats.append(mem_feats)
                rtv_mem_labels.append(mem_labels)

                ################################## 使用Triplet loss优化相似性，让检索能顺利进行
                # ---- 去掉False Positive的query
                pos_ids = q_target_labels >= 0
                if torch.sum(pos_ids) == 0:
                    continue
                pos_q_target_labels = q_target_labels[pos_ids]
                pos_sims = sims[pos_ids]

                # ---- 去掉不存在memory的Query
                nn_label_eq = torch.eq(pos_q_target_labels.reshape(-1, 1),
                                       k_labels.reshape(1, -1)).int()
                n_eq = torch.sum(nn_label_eq, dim=1)
                pos_ids = n_eq > 0
                pos_q_target_labels = pos_q_target_labels[pos_ids]
                pos_sims = pos_sims[pos_ids]
                if torch.sum(pos_ids) == 0:
                    continue

                # ---- 计算损失
                nn_label_eq = torch.eq(pos_q_target_labels.reshape(-1, 1),
                                       k_labels.reshape(1, -1)).int()
                # --- 与匹配上的GT类别内最相似的1个
                in_max_sims, in_max_indices = (pos_sims * nn_label_eq).max(dim=1)
                # --- 与匹配上的GT类别外最相似的1个
                out_max_sims, out_max_indices = (pos_sims * (1 - nn_label_eq)).max(dim=1)
                ######
                if torch.min(in_max_sims) < 0.1:
                    a = 0
                pos_dist = 1 - in_max_sims
                neg_dist = 1 - out_max_sims
                margin = 0.3
                nn_loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))
                nn_losses.append(nn_loss)

                # print('classes', list(name2id.keys()))
                # print('All gt_texts: ', set(sample.gt_instances.texts))
                # print('All rtv_texts: ', len(k_labels), set([id2name[int(l)] for l in k_labels]))
                # q_labels = all_query_labels[i]
                # print('pred_texts: ', len(q_labels), set([id2name[int(l)] for l in q_labels]))
                # print('nn_loss: ', nn_loss)
                # print('in_max_sims: ', len(in_max_sims), in_max_sims)
                # print('out_max_sims: ', len(out_max_sims), out_max_sims)

                # q_target_labels = all_query_target_labels[pos_ids]
                # pos_ids = q_target_labels >= 0
                # if torch.sum(pos_ids) > 0:
                #     pos_q = q[pos_ids]
                #     pos_target_labels = q_target_labels[pos_ids]
                #     pos_sims = sims[pos_q]
                #
                #     label_mask = torch.eq(pos_target_labels.reshape(-1, 1),
                #                           k_labels.reshape(1, -1)).int()
                #     pos_sims = pos_sims * label_mask
                #     q_max_sims, max_ids = sims.max(dim=1)
                #     # ---- 希望最大的余弦相似度接近1，保证每个q至少有一个同类别实例可以召唤出来



                # print('#' * 100)
                # sample = batch_data_samples[i]
                # print('classes', list(name2id.keys()))
                # print('All gt_texts: ', sample.gt_instances.texts)
                # print('All rtv_texts: ', len(k_labels), set([id2name[int(l)] for l in k_labels]))
                # q_labels = all_query_labels[i]
                # print('pred_texts: ', len(q_labels), [id2name[int(l)] for l in q_labels])
                # print('in_mem_texts: ', len(in_mem_labels), [id2name[int(l)] for l in in_mem_labels])
                # print('in_topk_sims: ', len(in_topk_sims), in_topk_sims)
                # print('out_mem_texts: ', len(out_mem_labels), [id2name[int(l)] for l in out_mem_labels])
                # print('out_topk_sims: ', len(out_topk_sims), out_topk_sims)

            if len(nn_losses) > 0:
                nn_losses = sum(nn_losses) / len(nn_losses)
            else:
                nn_losses = torch.Tensor([0.0]).float().to(device)
            losses['RTV_nn_loss'] = nn_losses * 3

            rtv_support_feats = []
            rtv_support_labels = []
            max_support_len = max([len(f) for f in rtv_mem_feats])
            for f, l in zip(rtv_mem_feats, rtv_mem_labels):
                rtv_box_f = self.box_support_mapping(f)
                if len(rtv_box_f) == max_support_len:
                    rtv_support_feats.append(torch.cat([rtv_box_f, base_support_feats]))
                    rtv_support_labels.append(torch.cat([l, base_support_labels]))
                    continue
                n_pad = max_support_len - len(rtv_box_f)
                pad_feats = torch.zeros([n_pad, rtv_box_f.shape[-1]]).float().to(device)
                pad_labels = torch.ones(n_pad).long().to(device) * -1

                rtv_support_feats.append(torch.cat([rtv_box_f, base_support_feats, pad_feats]))
                rtv_support_labels.append(torch.cat([l, base_support_labels, pad_labels]))


            rtv_support_feats = torch.stack(rtv_support_feats)
            rtv_support_labels = torch.stack(rtv_support_labels)

            rtv_losses = self.ict_head.loss(ict_x,
                                            batch_data_samples,
                                            rtv_support_feats,
                                            rtv_support_labels,
                                            rtv_support_labels,
                                            all_obj_embeds,
                                            all_obj_labels,
                                            **kwargs)
            for k, v in rtv_losses.items():
                losses[f'RTV_{k}'] = v
            ############# ------ 计算Retrieval损失，让同类的embeddings之间的距离大于非同类的
            box_p_support_feats = []
            box_p_support_labels = []
            max_support_len = max([len(f) for f in rtv_mem_feats])
            for f, l in zip(rtv_mem_feats, rtv_mem_labels):
                box_sem_f = self.box_prompt_head.embed_to_semantic(f)
                base_sem_f = self.box_visual_to_semantic(base_support_feats)

                if len(box_sem_f) == max_support_len:
                    box_p_support_feats.append(torch.cat([box_sem_f, base_sem_f]))
                    box_p_support_labels.append(torch.cat([l, base_support_labels]))
                    continue
                n_pad = max_support_len - len(f)
                pad_feats = torch.zeros([n_pad, box_sem_f.shape[-1]]).float().to(device)
                pad_labels = torch.ones(n_pad).long().to(device) * -1

                box_p_support_feats.append(torch.cat([box_sem_f, base_sem_f, pad_feats]))
                box_p_support_labels.append(torch.cat([l, base_support_labels, pad_labels]))

            box_p_support_feats = torch.stack(box_p_support_feats)
            box_p_support_labels = torch.stack(box_p_support_labels)

            box_prompt_losses = self.box_prompt_head.loss(box_x,
                                                          predictions,
                                                          batch_data_samples,
                                                          box_p_support_feats,
                                                          box_p_support_labels,
                                                          **kwargs)
            for k, v in box_prompt_losses.items():
                losses[f'BoxP_{k}'] = v

            ############# ----- Memory Bank更新（保存映射后的ict特征，BoxPrompt_Head -> ict_support_mapping）
            s_feats = local_memory_feats.detach()
            s_labels = local_memory_labels.detach()

            ### ----- 每个类别进行更新
            s_texts = [id2name[int(l)] for l in s_labels]
            s_uni_labels = [uni_name2id[name] for name in s_texts]
            s_uni_labels = torch.Tensor(s_uni_labels).long().to(device)
            with torch.no_grad():
                all_s_uni_labels = concat_all_gather_diff_size(s_uni_labels)
                all_s_feats = concat_all_gather_diff_size(s_feats)

                label_set = torch.unique(all_s_uni_labels)
                for l in label_set:
                    l = int(l)
                    in_cls_s_feats = all_s_feats[all_s_uni_labels == l]
                    in_cls_s_labels = all_s_uni_labels[all_s_uni_labels == l]

                    # 采样特征
                    ptr = int(self.queue_ptr[l])
                    sample_size = min([self.queue_len - ptr, len(in_cls_s_feats), self.q_max_update])
                    sample_indices = torch.randperm(len(in_cls_s_feats))[:sample_size].long().to(device)
                    sampled_feats = in_cls_s_feats[sample_indices]
                    sampled_labels = in_cls_s_labels[sample_indices]

                    self.queue[l, ptr:ptr + sample_size, :] = sampled_feats  # 更新特征
                    self.queue_label[l, ptr:ptr + sample_size] = sampled_labels  # 更新标签
                    self.queue_time[l, ptr:ptr + sample_size] = self.cur_iter  # 更新时间
                    ptr = (ptr + sample_size) % self.queue_len  # move pointer

                    self.queue_ptr[l] = ptr
                self.cur_iter = (self.cur_iter + 1) % self.max_iter

        return losses

    def loss_labelled(self,
                      x,
                      batch_data_samples,
                      support_data={},
                      support_name2id={},
                      support_type='visual',
                      dataset_flag=None):
        if support_type == 'visual':
            embed_name = 'visual_embeds'
            embed_mapping = self.visual_support_mapping
        else:
            embed_name = 'text_embeds'
            embed_mapping = self.text_support_mapping

        losses = dict()
        kwargs = dict()
        ##### ------------ Support：support feats是clip embeddings，默认为一个类
        num_classes = len(support_name2id)
        support_shot = int(np.random.randint(1, 5))
        support_feat_lens = [len(support_info[embed_name])
                             for support_info in support_data.values()]
        min_feat_len = min(support_feat_lens)
        support_shot = min([min_feat_len, support_shot])

        kwargs['support_shot'] = support_shot
        kwargs['num_classes'] = num_classes
        kwargs['num_in_classes'] = num_classes
        kwargs['align_style'] = 'labelled'
        ##### ------------ 设置Instance的标签
        start = 0
        device = x[0].device
        for i, sample in enumerate(batch_data_samples):
            bboxes = sample.gt_instances.bboxes
            texts = sample.gt_instances.texts
            labels = [support_name2id[cls_name] for cls_name in texts]

            ins_labels = torch.arange(start, start + len(bboxes)).to(device)
            sample.gt_instances['labels'] = torch.tensor(labels).to(device).long()
            sample.gt_instances['ins_labels'] = ins_labels

            start += len(bboxes)

        ##### ------------ 构造 视觉对齐的target
        dino_vis_embeds = [sample.gt_instances.visual_embeds.float() for sample in batch_data_samples]

        ######################## In-data的support构造
        support_feats = []
        support_labels = []
        real_support_shots = []

        for cls_name, support_info in support_data.items():
            cls_id = support_name2id[cls_name]
            feats = support_info[embed_name]
            labels = np.ones(len(feats)) * cls_id

            ids = np.arange(len(labels))
            sample_ids = np.random.permutation(ids)[:support_shot]
            real_support_shots.append(len(sample_ids))

            support_feats.append(feats[sample_ids])
            support_labels.append(labels[sample_ids])

        support_feats = torch.Tensor(np.concatenate(support_feats)).to(device)
        support_labels = torch.Tensor(np.concatenate(support_labels)).to(device)
        ######################## Out-data的support构造，label为-1，不参与检测器分类损失，只参与对齐损失
        max_neg_sample = int(np.random.randint(low=1, high=self.max_neg_sample))
        # neg_mapping = self.neg_mapping[dataset_flag]

        # --- 采样Out-data的负样本类别
        neg_classes_list = []

        for sample in batch_data_samples:
            pos_classes = set(list(sample.gt_instances.texts))
            neg_classes = []
            # ------ 合并所有的其他负样本，取交集
            # sample_neg_classes = [neg_mapping[pos_cls] for pos_cls in pos_classes]
            # neg_classes = reduce(lambda a, b: a.intersection(b), (set(lst) for lst in sample_neg_classes))
            # neg_classes = list(neg_classes)
            # if 'building' in neg_classes:
            #     neg_classes.extend(['building'] * 20)
            # # ----- 随机选取
            # neg_classes = np.random.permutation(neg_classes).tolist()[:max_neg_sample]
            # # ----- 去掉support中包含的元素，避免冲突
            # neg_classes = list(set(neg_classes) - set(support_data.keys()))
            neg_classes_list.append(neg_classes)
        # ----- batch内数量一致
        num_negs = [len(negs) for negs in neg_classes_list]
        max_neg_sample = min([max_neg_sample, *num_negs])
        neg_classes_list = [negs[:max_neg_sample] for negs in neg_classes_list]
        ####################################
        # for neg_class in neg_classes_list:
        #     a = deepcopy(list(support_data.keys()))
        #     a.extend(list(neg_class))
        #     print(a, ',')
        ####################################
        neg_support_feats_list = []
        neg_support_labels_list = []
        neg_support_slot_labels_list = []

        for neg_classes in neg_classes_list:
            neg_support_feats = []
            neg_support_labels = []
            neg_support_slot_labels = []
            cls_id = kwargs['num_classes']

            for cls_name in neg_classes:
                feats = self.uni_support_data[cls_name][embed_name]
                labels = np.ones(len(feats)) * cls_id

                ids = np.arange(len(labels))
                sample_ids = np.random.permutation(ids)[:support_shot]

                neg_support_feats.append(feats[sample_ids])
                # ----- 标签设置为-1
                neg_support_labels.append(labels[sample_ids] * 0 - 1)
                neg_support_slot_labels.append(labels[sample_ids])

                cls_id += 1
            if len(neg_support_feats) != 0:
                neg_support_feats = torch.Tensor(np.concatenate(neg_support_feats)).to(device)
                neg_support_labels = torch.Tensor(np.concatenate(neg_support_labels)).to(device).long()
                neg_support_slot_labels = torch.Tensor(np.concatenate(neg_support_slot_labels)).to(device).long()
                # print(neg_classes)
            else:
                neg_support_feats = torch.zeros([0, support_feats.shape[-1]]).to(device)
                neg_support_labels = torch.zeros(0).to(device)
                neg_support_slot_labels = torch.zeros(0).to(device)

            neg_support_feats_list.append(neg_support_feats)
            neg_support_labels_list.append(neg_support_labels)
            neg_support_slot_labels_list.append(neg_support_slot_labels)

        # --- 拼接正负样本
        support_feats_list = [torch.cat([support_feats, negs]) for negs in neg_support_feats_list]
        support_labels_list = [torch.cat([support_labels, negs]) for negs in neg_support_labels_list]
        support_slot_labels_list = [torch.cat([support_labels, negs]) for negs in neg_support_slot_labels_list]
        kwargs['num_classes'] += max_neg_sample
        kwargs['num_in_classes'] = kwargs['num_classes']
        ########################
        all_obj_embeds = dino_vis_embeds
        all_obj_labels = [sample.gt_instances.ins_labels for sample in batch_data_samples]  # 实例区分
        #############
        max_support_len = max([len(f) for f in support_feats_list])
        support_feats = []
        support_labels = []
        support_slot_labels = []
        for f, l, l2 in zip(support_feats_list, support_labels_list, support_slot_labels_list):
            if len(f) == max_support_len:
                support_feats.append(f)
                support_labels.append(l)
                support_slot_labels.append(l2)
                continue
            n_pad = max_support_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            support_feats.append(torch.cat([f, pad_feats]))
            support_labels.append(torch.cat([l, pad_labels]))
            support_slot_labels.append(torch.cat([l2, pad_labels]))

        support_feats = torch.stack(support_feats)
        support_labels = torch.stack(support_labels)
        support_slot_labels = torch.stack(support_slot_labels)

        support_feats = embed_mapping(support_feats)

        ##### ------------ 损失计算
        pred_cfg = dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.01,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)
        LB_losses, predictions = self.bbox_head.loss(x,
                                                     batch_data_samples,
                                                     support_feats,
                                                     support_labels,
                                                     support_slot_labels,
                                                     all_obj_embeds,
                                                     all_obj_labels,
                                                     ##########
                                                     return_predict=True,
                                                     pred_cfg=pred_cfg,
                                                     ##########
                                                     **kwargs)
        losses.update(LB_losses)
        if self.aux_bbox_head is not None:
            x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
            LB_losses_ex = self.aux_bbox_head.loss(x_aux,
                                                   batch_data_samples,
                                                   support_feats,
                                                   support_labels,
                                                   all_obj_embeds,
                                                   all_obj_labels,
                                                   **kwargs)
            for k, v in LB_losses_ex.items():
                losses['Ex_' + k] = v

        return losses, predictions

    def loss_pure_img(self,
                      x,
                      batch_data_samples):
        """
        纯图像的预训练：
        1. Support:              Visual Embeds
        2. Support类别：          聚类获得
        3. Object Embeddings:    [Visual embeds,]
        4. 对齐方式：              [Contrastive  ,]
        :param x:
        :param batch_data_samples:
        :return:
        """
        kwargs = dict()
        losses = dict()
        device = x[0].device

        ##### ------- 聚类获得标签, 1024 -> PCA -> 256 -> GMeans Cluster -> 20~40 class
        num_classes = []
        support_feats_list = []
        support_labels_list = []
        start = 0
        for sample in batch_data_samples:
            texts = np.array(sample.gt_instances.texts)
            embeds = sample.gt_instances.visual_embeds
            embeds = self.visual_support_mapping(embeds.to(device).float())
            text_set = set(list(sample.gt_instances.texts))
            n_class = len(text_set)
            n_embed = len(embeds)
            num_classes.append(n_class)

            labels = np.zeros(n_embed)
            support_feats = []
            support_labels = []
            for i, cls_name in enumerate(text_set):
                labels[texts == cls_name] = i
                cls_embed = torch.mean(embeds[texts == cls_name], dim=0)
                support_feats.append(cls_embed)
                support_labels.append(i)
            if len(support_feats) == 0:
                print(len(support_feats))
                print(sample.metainfo)
                a = 0
            support_feats = torch.stack(support_feats)
            support_labels = torch.tensor(support_labels).to(device).long()
            support_feats_list.append(support_feats)
            support_labels_list.append(support_labels)

            sample.gt_instances['labels'] = torch.tensor(labels).to(device).long()
            ins_labels = torch.arange(start, start + n_embed).to(device)
            sample.gt_instances['ins_labels'] = ins_labels

        ################################
        kwargs['support_shot'] = 1
        kwargs['num_classes'] = max(num_classes)
        kwargs['num_in_classes'] = max(num_classes)
        kwargs['align_style'] = 'pure_img'

        ##### ------------ Support：support feats是clip embeddings，默认为一个类
        ##### ------------ 构造 视觉对齐的target
        all_obj_embeds = [sample.gt_instances.visual_embeds for sample in batch_data_samples]
        all_obj_labels = [sample.gt_instances.ins_labels for sample in batch_data_samples]  # 实例区分
        ##### ------------ 对support
        max_support_len = max([len(f) for f in support_feats_list])
        support_feats = []
        support_labels = []
        for f, l in zip(support_feats_list, support_labels_list):
            if len(f) == max_support_len:
                support_feats.append(f)
                support_labels.append(l)
                continue
            n_pad = max_support_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            support_feats.append(torch.cat([f, pad_feats]))
            support_labels.append(torch.cat([l, pad_labels]))
        support_feats = torch.stack(support_feats)
        support_labels = torch.stack(support_labels)
        ##### ------------ 损失计算
        PI_losses = self.bbox_head.loss(x,
                                        batch_data_samples,
                                        support_feats,
                                        support_labels,
                                        support_labels,
                                        all_obj_embeds,
                                        all_obj_labels,
                                        **kwargs)
        losses.update(PI_losses)
        if self.aux_bbox_head is not None:
            x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
            PI_losses_ex = self.aux_bbox_head.loss(x_aux,
                                                   batch_data_samples,
                                                   support_feats,
                                                   support_labels,
                                                   all_obj_embeds,
                                                   all_obj_labels,
                                                   **kwargs)
            for k, v in PI_losses_ex.items():
                losses['Ex_' + k] = v
        return losses

    @torch.no_grad()
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True):
        if self.support_type in ['text', 'visual']:
            support_type = self.support_type
        elif self.support_type in ['random']:
            p = float(np.random.rand(1)[0])
            support_type = 'text' if p >= 0.5 else 'visual'
        else:
            raise Exception(f'Unrecognized support type {self.support_type}')

        if support_type == 'visual':
            embed_name = 'visual_embeds'
            embed_mapping = self.visual_support_mapping
        else:
            embed_name = 'text_embeds'
            embed_mapping = self.text_support_mapping
        x = self.extract_feat(batch_inputs)
        ##### ------------ 随机采样，构造support_feats
        kwargs = dict()
        ##### ------------ Support：support feats是clip embeddings，默认为一个类
        num_classes = len(self.val_name2id)
        support_shot = int(np.random.randint(3, 7))
        support_feat_lens = [len(support_info[embed_name])
                             for support_info in self.val_support_data.values()]
        min_feat_len = min(support_feat_lens)
        support_shot = min([min_feat_len, support_shot])

        kwargs['support_shot'] = support_shot
        kwargs['num_classes'] = num_classes
        kwargs['num_in_classes'] = num_classes
        kwargs['align_style'] = 'labelled'

        ##### ------------ 设置Instance的标签
        device = x[0].device
        support_feats = []
        support_labels = []
        for cls_name, support_info in self.val_support_data.items():
            cls_id = self.val_name2id[cls_name]
            feats = support_info[embed_name]
            labels = np.ones(len(feats)) * cls_id

            ids = np.arange(len(labels))
            sample_ids = np.random.permutation(ids)[:support_shot]

            support_feats.append(feats[sample_ids])
            support_labels.append(labels[sample_ids])

        support_feats = torch.Tensor(np.concatenate(support_feats)).to(device)
        support_labels = torch.Tensor(np.concatenate(support_labels)).to(device)
        ######### 需要根据support_labels进行排序，这样输出的cls_scores才是从0到C的cls_logit
        ######### 乱序的话结果就不对了
        sort_idx = torch.argsort(support_labels)
        support_labels = support_labels[sort_idx]
        support_feats = support_feats[sort_idx]

        support_feats = embed_mapping(support_feats)

        ##### ------------ 构造 Support的特征和label
        support_feats_list = [support_feats for i in range(len(batch_data_samples))]
        support_labels_list = [support_labels for i in range(len(batch_data_samples))]  # 实例区分

        max_support_len = max([len(f) for f in support_feats_list])
        support_feats = []
        support_labels = []
        for f, l in zip(support_feats_list, support_labels_list):
            if len(f) == max_support_len:
                support_feats.append(f)
                support_labels.append(l)
                continue
            n_pad = max_support_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            support_feats.append(torch.cat([f, pad_feats]))
            support_labels.append(torch.cat([l, pad_labels]))
        support_feats = torch.stack(support_feats)
        support_labels = torch.stack(support_labels)

        if self.val_using_aux:
            x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
            results_list = self.aux_bbox_head.predict(
                x_aux, batch_data_samples, support_feats, support_labels, rescale=rescale, **kwargs)
        else:
            results_list = self.bbox_head.predict(
                x, batch_data_samples, support_feats, support_labels, support_labels, rescale=rescale, **kwargs)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def catch_valid_ids_empty(self, valid_ids, sample):
        if len(valid_ids) == 0:
            print('#' * 100, '\n', 'Warning!!' * 10)
            print(f'{sample.img_path} Has no GTs besides SAM Object')
            print(f'Set Labels as 0 and use all objects')
            print('#' * 100)
            return True
        else:
            return False

    def prompt_extract_feats(self, batch_inputs):
        x = self.extract_feat(batch_inputs)
        return x

    def prompt_predict(self,
                       x,
                       batch_inputs: Tensor,
                       batch_data_samples: SampleList,
                       rescale: bool = True,
                       ###############
                       support_type='random',
                       support_shot=5,
                       val_support_data=dict(),
                       val_name2id=dict(),
                       val_using_aux=False,
                       ###############
                       output_embeds=False
                       ):
        #########################
        if support_type in ['text', 'visual']:
            pass
        elif support_type in ['random']:
            p = float(np.random.rand(1)[0])
            support_type = 'text' if p >= 0.5 else 'visual'
        else:
            raise Exception(f'Unrecognized support type {support_type}')

        if support_type == 'visual':
            embed_name = 'visual_embeds'
            embed_mapping = self.visual_support_mapping
        else:
            embed_name = 'text_embeds'
            embed_mapping = self.text_support_mapping
        ##### ------------ 随机采样，构造support_feats
        kwargs = dict()
        ##### ------------ Support：support feats是clip embeddings，默认为一个类
        num_classes = len(val_name2id)
        support_shot = int(support_shot)
        support_feat_lens = [len(support_info[embed_name])
                             for support_info in val_support_data.values()]
        min_feat_len = min(support_feat_lens)
        support_shot = min([min_feat_len, support_shot])

        kwargs['support_shot'] = support_shot
        kwargs['num_classes'] = num_classes
        kwargs['num_in_classes'] = num_classes
        kwargs['align_style'] = 'labelled'

        ##### ------------ 设置Instance的标签
        device = x[0].device
        support_feats = []
        support_labels = []
        for cls_name, support_info in val_support_data.items():
            cls_id = val_name2id[cls_name]
            feats = support_info[embed_name]
            labels = np.ones(len(feats)) * cls_id

            ids = np.arange(len(labels))
            sample_ids = np.random.permutation(ids)[:support_shot]

            support_feats.append(feats[sample_ids])
            support_labels.append(labels[sample_ids])

        support_feats = torch.Tensor(np.concatenate(support_feats)).to(device)
        support_labels = torch.Tensor(np.concatenate(support_labels)).to(device)
        ######### 需要根据support_labels进行排序，这样输出的cls_scores才是从0到C的cls_logit
        ######### 乱序的话结果就不对了
        sort_idx = torch.argsort(support_labels)
        support_labels = support_labels[sort_idx]
        support_feats = support_feats[sort_idx]

        support_feats = embed_mapping(support_feats)

        ##### ------------ 构造 Support的特征和label
        support_feats_list = [support_feats for i in range(len(batch_data_samples))]
        support_labels_list = [support_labels for i in range(len(batch_data_samples))]  # 实例区分

        max_support_len = max([len(f) for f in support_feats_list])
        support_feats = []
        support_labels = []
        for f, l in zip(support_feats_list, support_labels_list):
            if len(f) == max_support_len:
                support_feats.append(f)
                support_labels.append(l)
                continue
            n_pad = max_support_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            support_feats.append(torch.cat([f, pad_feats]))
            support_labels.append(torch.cat([l, pad_labels]))
        support_feats = torch.stack(support_feats)
        support_labels = torch.stack(support_labels)
        if output_embeds:
            x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
            results_list = self.aux_bbox_head.predict_with_embed(
                x_aux, batch_data_samples, support_feats, support_labels, rescale=rescale, **kwargs)
        else:
            if val_using_aux:
                x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
                results_list = self.aux_bbox_head.predict(
                    x_aux, batch_data_samples, support_feats, support_labels, rescale=rescale, **kwargs)
            else:
                results_list = self.bbox_head.predict(
                    x, batch_data_samples, support_feats, support_labels, support_labels, rescale=rescale, **kwargs)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def box_prompt_predict(self,
                           x,
                           batch_inputs: Tensor,
                           batch_data_samples: SampleList,
                           rescale: bool = True,
                           ###############
                           support_type='random',
                           support_shot=5,
                           val_support_data=dict(),
                           val_name2id=dict(),
                           val_using_aux=False,
                           ###############
                           output_embeds=False
                           ):
        #########################
        if support_type in ['text', 'visual']:
            pass
        elif support_type in ['random']:
            p = float(np.random.rand(1)[0])
            support_type = 'text' if p >= 0.5 else 'visual'
        else:
            raise Exception(f'Unrecognized support type {support_type}')

        if support_type == 'visual':
            embed_name = 'visual_embeds'
            embed_mapping = self.visual_support_mapping
        else:
            embed_name = 'text_embeds'
            embed_mapping = self.text_support_mapping
        ##### ------------ 随机采样，构造support_feats
        kwargs = dict()
        ##### ------------ Support：support feats是clip embeddings，默认为一个类
        num_classes = len(val_name2id)
        support_shot = int(support_shot)
        support_feat_lens = [len(support_info[embed_name])
                             for support_info in val_support_data.values()]
        min_feat_len = min(support_feat_lens)
        support_shot = min([min_feat_len, support_shot])

        kwargs['support_shot'] = support_shot
        kwargs['num_classes'] = num_classes
        kwargs['num_in_classes'] = num_classes
        kwargs['align_style'] = 'labelled'

        ##### ------------ 设置Instance的标签
        device = x[0].device
        support_feats = []
        support_labels = []
        for cls_name, support_info in val_support_data.items():
            cls_id = val_name2id[cls_name]
            feats = support_info[embed_name]
            labels = np.ones(len(feats)) * cls_id

            ids = np.arange(len(labels))
            sample_ids = np.random.permutation(ids)[:support_shot]

            support_feats.append(feats[sample_ids])
            support_labels.append(labels[sample_ids])

        support_feats = torch.Tensor(np.concatenate(support_feats)).to(device)
        support_labels = torch.Tensor(np.concatenate(support_labels)).to(device)
        ######### 需要根据support_labels进行排序，这样输出的cls_scores才是从0到C的cls_logit
        ######### 乱序的话结果就不对了
        sort_idx = torch.argsort(support_labels)
        support_labels = support_labels[sort_idx]
        support_feats = support_feats[sort_idx]

        support_feats = embed_mapping(support_feats)

        ##### ------------ 构造 Support的特征和label
        support_feats_list = [support_feats for i in range(len(batch_data_samples))]
        support_labels_list = [support_labels for i in range(len(batch_data_samples))]  # 实例区分

        max_support_len = max([len(f) for f in support_feats_list])
        support_feats = []
        support_labels = []
        for f, l in zip(support_feats_list, support_labels_list):
            if len(f) == max_support_len:
                support_feats.append(f)
                support_labels.append(l)
                continue
            n_pad = max_support_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            support_feats.append(torch.cat([f, pad_feats]))
            support_labels.append(torch.cat([l, pad_labels]))
        support_feats = torch.stack(support_feats)
        support_labels = torch.stack(support_labels)
        if output_embeds:
            x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
            results_list = self.aux_bbox_head.predict_with_embed(
                x_aux, batch_data_samples, support_feats, support_labels, rescale=rescale, **kwargs)
        else:
            if val_using_aux:
                x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
                results_list = self.aux_bbox_head.predict(
                    x_aux, batch_data_samples, support_feats, support_labels, rescale=rescale, **kwargs)
            else:
                results_list = self.bbox_head.predict(
                    x, batch_data_samples, support_feats, support_labels, support_labels, rescale=rescale, **kwargs)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def prompt_loss(self,
                    x,
                    batch_inputs: Tensor,
                    batch_data_samples: SampleList):
        losses = dict()
        ##### ------------- 统一名称
        for sample in batch_data_samples:
            texts = sample.gt_instances.texts
            texts_ = [self.norm_cls_map[cls_name] for cls_name in texts]
            sample.gt_instances.texts = texts_
            sample.cls_list = list(set([self.norm_cls_map[c] for c in sample.cls_list]))

        ##### ------------ 对Neck特征添加重建损失
        if self.with_image_rec_losses:
            loss_rec = self.rec_neck.loss(x, batch_inputs)
            losses['ImgRec_loss'] = loss_rec

        ##### ------------ 判断数据类型
        # dataset_flags = [sample.metainfo['dataset_flag']
        #                  for sample in batch_data_samples]
        # if len(set(dataset_flags)) != 1:
        #     raise Exception(f'Only one dataset is supported, but get: {dataset_flags}')
        # dataset_flag = dataset_flags[0]

        # print(f'dataset flag: {dataset_flag}')
        dataset_flag = None

        A = sum([x.view(-1)[0] for x in self.visual_support_mapping.parameters()])
        B = sum([x.view(-1)[0] for x in self.text_support_mapping.parameters()])
        x[0] = x[0] + A * 0.0 + B * 0.0
        ##### ------------ 分类讨论，进行forward
        ########### ------------ 构造support_data
        all_cls_names = []
        for sample in batch_data_samples:
            all_cls_names.extend(sample.cls_list)

        all_cls_names = set(all_cls_names)
        det_support_data = {k: v for k, v in self.uni_support_data.items() if k in all_cls_names}
        name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
        ########### ------------ 随机采样text或者visual prompt
        if self.support_type in ['text', 'visual']:
            support_type = self.support_type
        elif self.support_type in ['random']:
            p = float(np.random.rand(1)[0])
            support_type = 'text' if p >= 0.5 else 'visual'
        else:
            raise Exception(f'Unrecognized support type {self.support_type}')

        IT_losses = self.loss_labelled(x,
                                       batch_data_samples,
                                       support_data=det_support_data,
                                       support_name2id=name2id,
                                       support_type=support_type,
                                       dataset_flag=None)

        losses.update(IT_losses)
        ######################################################
        # predicts = self.prompt_predict(x,
        #                               batch_inputs,
        #                               deepcopy(batch_data_samples),
        #                               support_type='random',
        #                               support_shot=5,
        #                               val_support_data=det_support_data,
        #                               val_name2id=name2id,
        #                               val_using_aux=False)

        return losses

    def get_pred_results(self, results, id2name,
                         iou_thr=0.5,
                         score_thr=0.3):
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_pred_texts = []
        all_classes = []
        latent_classes = []
        for img_id in range(len(results)):
            sample = results[img_id]

            pred_boxes = sample.pred_instances.bboxes.detach()
            pred_labels = sample.pred_instances.labels.detach()
            pred_scores = sample.pred_instances.scores.detach()
            ####### NMS
            dets, keep_inds = nms_rotated(pred_boxes, pred_scores, iou_threshold=iou_thr)
            pred_boxes = pred_boxes[keep_inds]
            pred_scores = pred_scores[keep_inds]
            pred_labels = pred_labels[keep_inds]
            if keep_inds == None or torch.sum(keep_inds) == 0:
                all_pred_boxes.append([])
                all_pred_scores.append([])
                all_pred_labels.append([])
                all_pred_texts.append([])
                all_classes.extend([])
                latent_classes.append([])
                continue

            pred_texts = [id2name[int(l)] for l in pred_labels]
            latent_classes.append(sorted(list(set(pred_texts))))

            ####### Scores
            keep_inds = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep_inds]
            pred_scores = pred_scores[keep_inds]
            pred_labels = pred_labels[keep_inds]

            pred_texts = [id2name[int(l)] for l in pred_labels]
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)
            all_pred_texts.append(pred_texts)
            all_classes.extend(list(set(pred_texts)))
        all_classes = sorted(list(set(all_classes)))

        return all_pred_boxes, all_pred_scores, all_pred_labels, \
            all_pred_texts, all_classes, latent_classes

    def predict_with_supports(self,
                              batch_inputs: Tensor,
                              batch_data_samples: SampleList,
                              uni_support_data):
        return self.predict(batch_inputs, batch_data_samples)
        #### ----- Alignment Head预测
        # feat_x = list(self.extract_feat(batch_inputs))
        # det_support_data = deepcopy(uni_support_data)
        # name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
        # id2name = {cat_id: name for cat_id, name in enumerate(det_support_data.keys())}
        # align_results = self.prompt_predict(
        #     deepcopy(feat_x),
        #     batch_inputs,
        #     deepcopy(batch_data_samples),
        #     val_support_data=det_support_data,
        #     support_shot=32,
        #     val_name2id=name2id,
        #     val_using_aux=True,
        #     rescale=False
        # )
        #
        # align_pred_boxes, align_pred_scores, align_pred_labels, \
        #     align_pred_texts, align_classes, latent_classes = \
        #     self.get_pred_results(align_results, id2name,
        #                           iou_thr=0.1, score_thr=0.3)
        # if len(align_classes) != 0:
        #     ###### ----------- Fusion Head
        #     det_support_data = {k: v
        #                         for k, v in uni_support_data.items() if k in align_classes}
        #     name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
        #     id2name = {cat_id: name for cat_id, name in enumerate(det_support_data.keys())}
        #
        #     fusion_results = self.prompt_predict(
        #         deepcopy(feat_x),
        #         batch_inputs,
        #         deepcopy(batch_data_samples),
        #         val_support_data=det_support_data,
        #         val_name2id=name2id,
        #         val_using_aux=False,
        #         rescale=False,
        #         support_shot=10,
        #     )
        #
        #     fusion_pred_boxes, fusion_pred_scores, fusion_pred_labels, \
        #         fusion_pred_texts, fusion_classes, _ = \
        #         self.get_pred_results(fusion_results, id2name,
        #                               iou_thr=0.1, score_thr=0.3)
        # else:
        #     fusion_pred_boxes = []
        #     fusion_pred_scores = []
        #     fusion_pred_labels = []
        #     fusion_pred_texts = []
        #     fusion_classes = []
        #
        # return fusion_pred_boxes, fusion_pred_scores, fusion_pred_labels, fusion_pred_texts, fusion_classes

    ##################################################################
    def get_embed_preds(self,
                        x,
                        batch_data_samples: SampleList):
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        x_aux = [self.aux_convs[i](f) for i, f in enumerate(x)]
        bbox_preds, angle_preds, pred_embeds = \
            self.aux_bbox_head.forward_wo_supports(x_aux)
        box_prompt_out = self.aux_bbox_head.get_pos_embeds(
            [],
            bbox_preds,
            angle_preds,
            pred_embeds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        (new_batch_gt_instances, all_pos_pred_embeds,
         all_pos_pred_boxes, all_pos_pred_ins_labels, all_pos_pred_sem_embeds) = box_prompt_out

        return box_prompt_out

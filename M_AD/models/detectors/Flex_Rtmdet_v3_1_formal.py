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
"""
让v3_1更合理一些：
1. 使用sample.cls_list和统一的uni_support来构造batch的support_labels和feats
    1. 目的：利用上FederatedLabels
2. 使用normalized_class_dict将各个类别映射到统一名称
    1. 目的：避免negative和positive出现相同名称，导致错误
3. neg_support_data改为uni_support_data，并且不用额外载入，直接使用多数据集的support构造
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
                 num_val_prompts=0,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

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
                if k not in self.norm_cls_map.keys():
                    self.norm_cls_map[k] = k
                    print(f'Add New key in norm_cls_map: {k}')
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
            aux_channels = [self.bbox_head.feat_channels,] * 3
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
        self.num_val_prompts = num_val_prompts


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
        x = list(self.extract_feat(batch_inputs))
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

        # print(f'dataset flag: {dataset_flag}')

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
            # print(f'Num Pos: ', len(set(pos_cls_names)), 'Num Neg: ', len(neg_cls_names))
            name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
            # if dataset_flag == 'Data31_HardResampling':
            #     pos_names = sorted(list(set(pos_cls_names)))
            #     neg_names = sorted(list(set(neg_cls_names)))
            #     print(f'Pos: {len(pos_names)}', pos_names)
            #     print(f'Neg: {len(neg_names)}', neg_names)

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
                                           dataset_flag=dataset_flag)

            losses.update(IT_losses)
        # if dataset_flag in ['D4_HRRSD', 'D6_Xview']:
        #     for k, v in losses.items():
        #         if 'bbox' in k:
        #             # print(f'got {dataset_flag}, set loss zero')
        #             if type(v) in [tuple, list]:
        #                 losses[k] = [i * 0.0 for i in v]
        #             else:
        #                 losses[k] = v * 0.0
        #             # print(losses)

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
        LB_losses = self.bbox_head.loss(x,
                                        batch_data_samples,
                                        support_feats,
                                        support_labels,
                                        support_slot_labels,
                                        all_obj_embeds,
                                        all_obj_labels,
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

        return losses

    def loss_img_text(self,
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
                                                   support_labels,
                                                   all_obj_embeds,
                                                   all_obj_labels,
                                                   **kwargs)
            for k, v in PI_losses_ex.items():
                losses['Ex_' + k] = v
        return losses

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
        if self.num_val_prompts == 0:
            support_shot = int(np.random.randint(3, 7))
        else:
            support_shot = self.num_val_prompts

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
        support_labels_list = [support_labels for i in range(len(batch_data_samples))]     # 实例区分

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
        support_labels_list = [support_labels for i in range(len(batch_data_samples))]     # 实例区分

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


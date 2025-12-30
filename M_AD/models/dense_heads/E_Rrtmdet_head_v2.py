# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmcv.cnn import ConvModule, Scale, is_norm
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn
import numpy as np

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb
from mmengine.model import BaseModule
from mmcv.cnn.bricks import build_norm_layer

import torch.nn.functional as F
from mmrotate.models.dense_heads.rotated_rtmdet_head import RotatedRTMDetHead, RotatedRTMDetSepBNHead
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                                unpack_gt_instances)
from mmdet.structures import OptSampleList, SampleList
from copy import deepcopy
import math

from mmdet.models.utils import (images_to_levels, multi_apply, sigmoid_geometric_mean,
                                unmap)

"""
纯预训练
"""

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BaseClsHead(nn.Module):

    def __init__(self):
        super().__init__()

    def get_matching_scores(self,
                            matching_scores,
                            support_labels,
                            **kwargs):
        """

        :param matching_scores: B, N, M
        :param support_labels:  B, M
        :param kwargs:
        :return:
        """
        # ----- padding的score设置为-inf
        B, N, M = matching_scores.shape
        support_labels = support_labels[:, None, :].expand([B, N, M])
        matching_scores[support_labels < 0] = -10 # float('-inf')

        # ----- 只有两种情况，相差为1，包括一个out_class，或者相差为0，不包括out_class
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        support_shot = kwargs['support_shot']
        assert (num_classes - num_in_classes) in [0, 1]

        len_in_classes = num_in_classes * support_shot
        cls_scores = torch.full([B, N, num_classes],
                                float('-inf'),
                                device=matching_scores.device)
        # ----- in_class的分类分数，由于每个类数量是一致的，因此直接取max
        in_match_scores = (matching_scores[:, :, :len_in_classes].
                           reshape(B, N, num_in_classes, support_shot))
        in_cls_scores = torch.max(in_match_scores, dim=-1)[0]
        cls_scores[:, :, :num_in_classes] = in_cls_scores

        # ----- out_class的分类分数，也是取max
        if M > len_in_classes:
            out_match_scores = matching_scores[:, :, len_in_classes:]
            out_cls_scores = torch.max(out_match_scores, dim=-1)[0]
            cls_scores[:, :, -1] = out_cls_scores

        return cls_scores

class ContrastiveEmbed(BaseClsHead):

    def __init__(self):
        super().__init__()
        # ------------ 结果缩放
        self.log_scale_t1 = nn.Parameter(
            torch.Tensor([-float(1.0)]), requires_grad=True)
        self.bias_t1 = nn.Parameter(
            torch.Tensor([-float(4.0)]), requires_grad=True)

    def forward(self,
                pred_embeds,
                support_feats,
                support_labels,
                visual_fc,
                text_fc,
                **kwargs
                ) -> (Tensor, Tensor):
        # ----- Semantic mapping
        B, D, H, W = pred_embeds.shape
        x = (pred_embeds.permute(0, 2, 3, 1).reshape(B, H * W, D) +
             sum(x.view(-1)[0]for x in self.parameters()) * 0.)
        align_style = kwargs['align_style']
        if align_style in ['labelled', 'pure_img']:             # ---- support是visual embeds
            bias = self.bias_t1
            log_scale = self.log_scale_t1
            x = visual_fc(x)
            support_feats = visual_fc(support_feats)
        else:
            raise Exception(f'Unknown alignment style {align_style}')

        w = F.normalize(support_feats, dim=-1)
        # ----- Get matching scores: B N D x B M D -> B N M
        match_logit = x @ w.transpose(-1, -2)
        scaled_logit = match_logit * log_scale.exp() + bias
        # ----- Get class logits: B N M -> B N C -> B C H W
        cls_logits = self.get_matching_scores(scaled_logit, support_labels, **kwargs)
        cls_logits = (cls_logits.reshape(B, H, W, cls_logits.shape[-1]).
                      permute(0, 3, 1, 2).contiguous())

        return cls_logits

from M_AD.models.utils.transformer_modular_yolo import TwoWayTransformerModularYOLO

class DummyCrossAttention(nn.Module):
    def __init__(
            self) -> None:
        super().__init__()

    def forward(self, queries, keys):
        return queries, keys

@MODELS.register_module()
class OpenRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """
    提高并行性，增加推理速度

    结论：
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 with_obj_align=False,
                 ## ---
                 with_slot_embed=True,
                 cross_mlp_dim=2048,
                 cross_num_layers=3,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        super().__init__(*args, **kwargs)

        self.with_obj_align = with_obj_align
        self.text_fc = nn.Identity()
        self.visual_fc = nn.Sequential(
            nn.Linear(embed_dims, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, embed_dims)
        )
        # --------- 显式的对齐
        if self.with_obj_align:
            raise Exception(f'with_obj_align has been removed')
        ###########################
        self.cross_input = nn.Linear(self.feat_channels, embed_dims)
        self.cross_norm = nn.LayerNorm(embed_dims)
        self.cross_attention = TwoWayTransformerModularYOLO(embedding_dim=embed_dims,
                                                            num_heads=8,
                                                            mlp_dim=cross_mlp_dim,
                                                            with_query_self_attn=False,   # 不利于拓展
                                                            with_cross_query_to_key=True,
                                                            with_cross_key_to_query=True,
                                                            depth=cross_num_layers)
        self.cross_output = nn.Linear(embed_dims, self.feat_channels)
        ###########################
        self.with_slot_embed = with_slot_embed
        if self.with_slot_embed:
            self.class_slot_embeds = (
                nn.Embedding(num_embeddings=256, embedding_dim=embed_dims))

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.rtm_cls = nn.ModuleList()
        self.rtm_cls_heads = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.embed_dims,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_cls_heads.append(ContrastiveEmbed())


    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()

    def loss(self,
             x: Tuple[Tensor],
             batch_data_samples: SampleList,
             support_feats,
             support_labels,
             obj_embeds_list,
             obj_labels_list,
             **kwargs) -> dict:
        # len(support_feats_list[0]) = support_shot * num_in_classes + num_out_instance(或许没有)
        device = x[0].device
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        support_shot = kwargs['support_shot']
        ###          ---- 对support进行padding
        max_obj_embed_len = max([len(e) for e in obj_embeds_list])
        obj_embeds = []
        obj_labels = []
        for f, l in zip(obj_embeds_list, obj_labels_list):
            if len(f) == max_obj_embed_len:
                obj_embeds.append(f)
                obj_labels.append(l)
                continue
            n_pad = max_obj_embed_len - len(f)
            pad_feats = torch.zeros([n_pad, f.shape[-1]]).float().to(device)
            pad_labels = torch.ones(n_pad).long().to(device) * -1
            obj_embeds.append(torch.cat([f, pad_feats]))
            obj_labels.append(torch.cat([l, pad_labels]))
        obj_embeds = torch.stack(obj_embeds)
        obj_labels = torch.stack(obj_labels)

        ###################################
        losses = dict()
        outs = self(x, support_feats, support_labels, obj_embeds, **kwargs)
        out_obj_embeds = outs[-1]
        out_support_feats = outs[-2]
        outs = outs[:-2]

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore,
                              out_obj_embeds, obj_labels)
        losses_head = self.loss_by_feat(*loss_inputs, **kwargs)
        losses.update(losses_head)

        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                support_feats,
                support_labels,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        device = x[0].device
        ###############
        outs = self(x, support_feats, support_labels, **kwargs)
        out_support_feats = outs[-1]
        pred_embeds = outs[-2]
        outs = outs[:-2]
        ###############
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self,
                feats: Tuple[Tensor, ...],
                support_feats,
                support_labels,
                obj_embeds=None,
                **kwargs) -> tuple:
        device = feats[0].device
        ################# support feats 和 obj_embeddings进行映射
        out_obj_embeds = obj_embeds
        #######################
        spatial_shapes = [f.shape[2:] for f in feats]
        flatten_feats = torch.cat([f.flatten(2) for f in feats], dim=-1).permute(0, 2, 1)

        B, N, C = flatten_feats.shape
        in_flatten_feats = self.cross_norm(self.cross_input(flatten_feats))
        ########### ----- 建立label到[0, n_label]的随机映射，并获得position embeddings
        ########### 2, 2, 1, 1, 0, 0 -> [1, 2, 0] -> 1, 1, 0, 0, 2, 2
        if self.with_slot_embed:
            label_set = torch.unique(support_labels).detach().cpu().numpy().astype(np.int64)
            label_set = np.random.permutation(label_set).tolist()
            permuted_labels = torch.zeros_like(support_labels).long().to(device)
            for p_l, l in enumerate(label_set):
                permuted_labels[support_labels == l] = p_l
            permuted_labels = permuted_labels.contiguous()
            slot_embeds = self.class_slot_embeds(permuted_labels)
            support_feats = support_feats + slot_embeds

        out_support_feats, out_flatten_feats = self.cross_attention(support_feats, in_flatten_feats)
        out_flatten_feats = self.cross_output(out_flatten_feats)
        out_flatten_feats = out_flatten_feats.permute(0, 2, 1)
        out_feats = []
        start = 0
        for spatial_shape in spatial_shapes:
            H, W = spatial_shape
            out_feats.append(out_flatten_feats[:, :, start: start + H*W].
                                   reshape(B, C, H, W))
            start += H*W
        #######################
        pred_embeds = []
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, stride) in enumerate(
                zip(out_feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            pred_embed = self.rtm_cls[idx](cls_feat)
            cls_logit = self.rtm_cls_heads[idx](pred_embed,
                                                out_support_feats,
                                                support_labels,
                                                visual_fc=self.visual_fc,
                                                text_fc=self.text_fc,
                                                **kwargs)
            cls_score = cls_logit
            ########################################
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)


            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
            pred_embeds.append(pred_embed)
        if out_obj_embeds is not None:
            return (tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(pred_embeds),
                    out_support_feats, out_obj_embeds)
        else:
            return (tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(pred_embeds),
                    out_support_feats)

    ############## --------------- Loss 计算相关 ---------------
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int],
                            **kwargs):
        num_classes = kwargs['num_classes']
        num_in_classes = kwargs['num_in_classes']
        #######################################
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_out_channels = cls_score.shape[1]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, cls_out_channels).contiguous()

        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # ---- 如果包含SAM Obj，则num_classes=21, num_in_classes=20, cls_score的长度为21，sam_obj的label为20, bg为21
        bg_class_ind = num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0

        return (loss_cls, loss_bbox, loss_angle, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     embed_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     obj_embeds=None,
                     obj_labels=None,
                     **kwargs):
        losses = dict()
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_out_channels = cls_scores[0].shape[1]
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, ins_labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets
        ##############################################################################
        ############## 计算object embeddings的对齐loss
        # ----- n_level x [(B, D, H_i, W_i)] -> B x n_position x D -> N x D
        flatten_embeds = torch.cat([
            embed.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                              self.embed_dims)
            for embed in embed_preds
        ], 1)
        # ----- n_level x [(B, H_i*W_i)] -> B x n_position -> N
        flatten_ins_labels = torch.cat(ins_labels_list, dim=1)

        flatten_embeds = flatten_embeds.reshape(-1, self.embed_dims)
        flatten_ins_labels = flatten_ins_labels.reshape(-1)

        # ----- Positive的预测embeddings
        pos_pred_embeds = flatten_embeds[flatten_ins_labels >= 0]
        # pos_ins_labels: [0, 0, 1, 2, 0, 0, 2, 4]，代表每个pred_embeds对应的instance类别
        pos_ins_labels = flatten_ins_labels[flatten_ins_labels >= 0]

        target_embeds = obj_embeds.reshape(-1, obj_embeds.shape[-1])[obj_labels.reshape(-1) >= 0]
        obj_ins_labels = [instance.ins_labels for instance in batch_gt_instances]
        obj_ins_labels = torch.cat(obj_ins_labels)
        assert len(target_embeds) == len(obj_ins_labels)
        # ----- 计算对齐损失
        if self.with_obj_align:
            loss_align = self.obj_align_branch(pos_pred_embeds,
                                               target_embeds,
                                               pos_ins_labels,
                                               obj_ins_labels,
                                               visual_fc=self.visual_fc,
                                               text_fc=self.text_fc,
                                               **kwargs)
            losses.update(loss_align)
            # losses['InsAln_Psims'] = pos_sims.detach()
            # losses['InsAln_NSims'] = neg_sims.detach()
        ##############################################################################

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes
        # ----loss_by_feat_single对每个层级的特征单独计算特征
        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
            self.loss_by_feat_single, cls_scores, decoded_bboxes,
            angle_preds_list, labels_list, label_weights_list,
            bbox_targets_list, assign_metrics_list,
            self.prior_generator.strides, **kwargs)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            losses.update(dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle))
        else:
            losses.update(dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox))
        return losses

    def get_targets(self,
                    cls_scores: Tensor,
                    bbox_preds: Tensor,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True):
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        ######## ------------- 每张图片单独计算target，返回了实例标签
        (all_anchors, all_labels, all_ins_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics, sampling_results_list) = multi_apply(
            self._get_targets_single,
            cls_scores.detach(),
            bbox_preds.detach(),
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        ins_labels_list = images_to_levels(all_ins_labels, num_level_anchors)

        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics,
                                               num_level_anchors)

        return (anchors_list, labels_list, ins_labels_list, label_weights_list,
                bbox_targets_list, assign_metrics_list, sampling_results_list)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        """
        sampling_result.pos_assigned_gt_inds: 每个pos分配到的gt, N_pos
        sampling_result.pos_inds：pos的anchor的id, N_pos
        sampling_result.neg_inds：neg的anchor的id, N_neg
        sampling_result.pos_is_gt: 可以忽略，因为sampling是一个伪采样器

        """

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors,),
                                  cls_scores.shape[-1],
                                  dtype=torch.long)
        ################ instance的标签
        ins_labels = anchors.new_full((num_valid_anchors,),
                                      -1,
                                      dtype=torch.long)
        ################
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(
                self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            ################
            ins_gt_labels = gt_instances.ins_labels
            ins_labels[pos_inds] = ins_gt_labels[sampling_result.pos_assigned_gt_inds]
            ################
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # ---- 分配上的gt（理应每个gt都被分配到）
        # ---- assign_metrics中只有pos的>0，neg=0
        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=cls_scores.shape[-1])
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, ins_labels, label_weights, bbox_targets, assign_metrics,
                sampling_result)

    ################################ Predict #################################

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_out_channels = cls_score.shape[0]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

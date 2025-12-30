# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig

from typing import Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
import mmrotate.registry as mmr_registry
from M_AD.models.roi_heads.bbox_heads.Ace_convfc_bbox_head import AceConvFCBBoxHead
import torch
import torch.nn.functional as F
from copy import deepcopy

@mmr_registry.MODELS.register_module()
class HinFCBBoxHead(BBoxHead):
    """
    num_cls_fcs=2,
    num_reg_convs=0,
    num_reg_fcs=2,
    """

    def __init__(self,
                 *args,
                 reg_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 cls_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 **kwargs) -> None:
        super().__init__(*args,
                         reg_predictor_cfg=reg_predictor_cfg,
                         cls_predictor_cfg=cls_predictor_cfg,
                         **kwargs)

        self.fc_embed = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

        box_dim = self.bbox_coder.encode_size
        self.fc_reg = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, box_dim)
        )

        #############
        self.fc_cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,256)
        )


        self.bg = nn.Parameter(
            torch.Tensor([float(1.0)]), requires_grad=True)
        self.log_scale_t1 = nn.Parameter(
            torch.Tensor([float(2.0)]), requires_grad=True)
        self.bias_t1 = nn.Parameter(
            torch.Tensor([-float(1.0)]), requires_grad=True)

    def extract_embed(self, x):
        embed_pred = self.fc_embed(x)
        return embed_pred

    # def forward(self,
    #             x,
    #             batch_ids,
    #             support_feats, support_labels,
    #             **kwargs) -> tuple:
    #     """
    #
    #     :param x: (N, 256, 7, 7)
    #     :param batch_ids: N, 表示x属于哪一个batch
    #     :param support_feats:
    #     :param support_labels:
    #     :param kwargs:
    #     :return:
    #     """
    #     # separate branches
    #     x_cls = x
    #     x_reg = x
    #
    #     embed_pred = self.fc_embed(x_cls)
    #     bbox_pred = self.fc_reg(x_reg)
    #
    #     bias = self.bias_t1
    #     log_scale = self.log_scale_t1
    #     box_sem_feats = self.fc_cls(embed_pred)
    #     # ----- 这里w已经转换到了semantic空间，和visual_support等特征合并到了一起，因此不用经过fc_cls
    #     support_feats = F.normalize(support_feats, dim=-1)
    #
    #     cls_scores = []
    #     # ----- 每张图片的support都不一样，需要单独考虑
    #     for i in range(len(support_feats)):
    #         in_batch_ids = batch_ids == i
    #         x = box_sem_feats[in_batch_ids]
    #         w = support_feats[i]
    #         match_logit = x @ w.transpose(-1, -2)
    #         scaled_logit = match_logit * log_scale.exp() + bias
    #
    #         cls_logits = self.get_matching_scores(scaled_logit,
    #                                               support_labels[i], **kwargs)
    #         cls_scores.append(cls_logits)
    #     cls_scores = torch.cat(cls_scores).contiguous()
    #     # print(cls_scores.shape, bbox_pred.shape)
    #
    #     return cls_scores, bbox_pred
    #
    # def get_matching_scores(self,
    #                         matching_scores,
    #                         support_labels,
    #                         **kwargs):
    #     # ----- num_classes是定义的类别个数，support_labels在-1 ~ num_classes-1之间, -1表示忽略的support，或者是负样本，或者是padding特征
    #     N, M = matching_scores.shape
    #     num_classes = kwargs['num_classes']
    #     # ----- 设置与-1的support之间的相似度为-100，不计入损失
    #     # ----- -1无法用来进行scatter_reduce，由于support_labels实际上代表了index，因此将-1映射为num_classes
    #     valid_support_labels = deepcopy(support_labels)
    #     valid_support_labels[valid_support_labels < 0] = num_classes
    #     valid_support_labels = valid_support_labels[None, :].expand([N, M])
    #
    #     # ----- cls_scores最后一列为-1的元素
    #     cls_scores = torch.full((N, num_classes + 1), float(-100), device=matching_scores.device)
    #     # ----- 进行scatter_reduce，获取每一类的分数，并去掉最后一列分数
    #     cls_scores = cls_scores.scatter_reduce(-1, valid_support_labels.long(), matching_scores, reduce="amax")
    #     # ----- 最后一个元素替换为可学习的bg参数
    #     bg_scores = self.bg[None].expand([N, 1])
    #     out_cls_scores = torch.cat([cls_scores[:, :-1], bg_scores], dim=-1)
    #
    #     return out_cls_scores

    def forward(self,
                x,
                batch_ids,
                support_feats, support_labels,
                **kwargs) -> tuple:
        """

        :param x: (N, 256, 7, 7)
        :param batch_ids: N, 表示x属于哪一个batch
        :param support_feats:
        :param support_labels:
        :param kwargs:
        :return:
        """
        # separate branches
        x_cls = x
        x_reg = x

        embed_pred = self.fc_embed(x_cls)
        bbox_pred = self.fc_reg(x_reg)

        bias = self.bias_t1
        log_scale = self.log_scale_t1
        box_sem_feats = self.fc_cls(embed_pred)
        # ----- 这里w已经转换到了semantic空间，和visual_support等特征合并到了一起，因此不用经过fc_cls
        support_feats = F.normalize(support_feats, dim=-1)

        # ----- 将x转换成基于batch的，进行padding
        device = embed_pred.device

        batch_box_sem_feats = []
        max_len = max([torch.sum(batch_ids == i) for i in range(len(support_feats))])
        for i in range(len(support_feats)):
            in_batch_ids = batch_ids == i
            in_batch_feats = box_sem_feats[in_batch_ids]
            if len(in_batch_feats) == max_len:
                batch_box_sem_feats.append(in_batch_feats)
                continue
            n_pad = max_len - len(in_batch_feats)
            pad_feats = torch.zeros([n_pad, in_batch_feats.shape[-1]]).float().to(device)
            batch_box_sem_feats.append(torch.cat([in_batch_feats, pad_feats]))
        batch_box_sem_feats = torch.stack(batch_box_sem_feats)

        # ----- 计算分类分数
        x = batch_box_sem_feats
        w = support_feats
        match_logit = x @ w.transpose(-1, -2)
        scaled_logit = match_logit * log_scale.exp() + bias

        cls_logits = self.get_matching_scores(scaled_logit,
                                              support_labels, **kwargs)
        # ----- 按照batch还原，去除padding
        cls_scores = []
        for i in range(len(support_feats)):
            in_batch_ids = batch_ids == i
            n_valid = torch.sum(in_batch_ids)
            cls_scores.append(cls_logits[i, :n_valid])
        cls_scores = torch.cat(cls_scores)

        return cls_scores, bbox_pred

    def get_matching_scores(self,
                            matching_scores,
                            support_labels,
                            **kwargs):
        # ----- num_classes是定义的类别个数，support_labels在-1 ~ num_classes-1之间, -1表示忽略的support，或者是负样本，或者是padding特征
        B, N, M = matching_scores.shape
        num_classes = kwargs['num_classes']
        # ----- 设置与-1的support之间的相似度为-100，不计入损失
        # ----- -1无法用来进行scatter_reduce，由于support_labels实际上代表了index，因此将-1映射为num_classes
        valid_support_labels = deepcopy(support_labels)
        valid_support_labels[valid_support_labels < 0] = num_classes
        valid_support_labels = valid_support_labels[:, None, :].expand([B, N, M])

        # ----- cls_scores最后一列为-1的元素
        cls_scores = torch.full((B, N, num_classes + 1), float(-100), device=matching_scores.device)
        # ----- 进行scatter_reduce，获取每一类的分数，并去掉最后一列分数
        cls_scores = cls_scores.scatter_reduce(-1, valid_support_labels.long(), matching_scores, reduce="amax")
        # ----- 最后一个元素替换为可学习的bg参数
                # ----- 最后一个元素替换为可学习的bg参数
        bg_scores = self.bg[None, None].expand([B, N, 1])
        out_cls_scores = torch.cat([cls_scores[:, :, :-1], bg_scores], dim=-1)

        return out_cls_scores

    #################################### 适应到开放词汇的改动 #####################################################

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict, **kwargs) -> tuple:
        num_classes = kwargs['num_classes']
        ######
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True,
                    **kwargs) -> tuple:
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg,
            **kwargs)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None,
                        **kwargs) -> dict:

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat, **kwargs)
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override, **kwargs)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)



    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None,
             **kwargs) -> dict:
        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = kwargs['num_classes']
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), kwargs['num_classes'],
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses


    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        **kwargs) -> InstanceList:

        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg,
                **kwargs)
            result_list.append(results)

        return result_list
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None,
            **kwargs) -> InstanceData:
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=kwargs['num_classes'],
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else kwargs['num_classes']
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results


    def refine_bboxes(self, sampling_results: Union[List[SamplingResult],
                                                    InstanceList],
                      bbox_results: dict,
                      batch_img_metas: List[dict],
                      **kwargs) -> InstanceList:

        num_classes = kwargs['num_classes']

        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        labels = bbox_results['bbox_targets'][0]
        cls_scores = bbox_results['cls_score']
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        if self.custom_activation:
            # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
            cls_scores = self.loss_cls.get_activation(cls_scores)
        if cls_scores.numel() == 0:
            return None
        if cls_scores.shape[-1] == num_classes + 1:
            # remove background class
            cls_scores = cls_scores[:, :-1]
        elif cls_scores.shape[-1] != num_classes:
            raise ValueError('The last dim of `cls_scores` should equal to '
                             '`num_classes` or `num_classes + 1`,'
                             f'but got {cls_scores.shape[-1]}.')
        labels = torch.where(labels == num_classes, cls_scores.argmax(1),
                             labels)

        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            results = InstanceData(bboxes=bboxes[keep_inds.type(torch.bool)])
            results_list.append(results)

        return results_list

    def regress_by_class(self, priors: Tensor, label: Tensor,
                         bbox_pred: Tensor, img_meta: dict) -> Tensor:
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            priors (Tensor): Priors from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        reg_dim = self.bbox_coder.encode_size
        if not self.reg_class_agnostic:
            label = label * reg_dim
            inds = torch.stack([label + i for i in range(reg_dim)], 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size()[1] == reg_dim

        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape)
        return regressed_bboxes















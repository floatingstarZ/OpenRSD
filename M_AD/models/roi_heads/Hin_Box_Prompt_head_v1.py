# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
import mmrotate.registry as mmr_registry
from M_AD.models.roi_heads.Ace_cascade_roi_head import AceCascadeRoIHead

@mmr_registry.MODELS.register_module()
class HinBoxPromptHead(AceCascadeRoIHead):
    """
    Cascade RoI Head, with Box Prompt Extractor
    """
    def __init__(self,
                 *args,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)

    def embed_to_semantic(self, embed_pred):
        stage = 0
        bbox_head = self.bbox_head[stage]
        embed_sem = bbox_head.fc_cls(embed_pred)
        return embed_sem

    def extract_embed(self, x, proposals):
        stage = 0
        rois = bbox2roi(proposals)

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        embed_pred = bbox_head.extract_embed(bbox_feats)

        n_img = len(proposals)
        batch_ids = rois[:, 0]
        object_embeds = []
        for i in range(n_img):
            in_batch_ids = batch_ids == i
            object_embeds.append(embed_pred[in_batch_ids])

        return object_embeds

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor,
                      support_feats, support_labels, **kwargs) -> dict:
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        batch_ids = rois[:, 0]
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats, batch_ids, support_feats, support_labels, **kwargs)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  support_feats, support_labels, **kwargs) -> dict:
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois,
                                          support_feats, support_labels, **kwargs)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage],
            **kwargs)
        bbox_results.update(bbox_loss_and_target)

        return bbox_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList, support_feats, support_labels, **kwargs) -> dict:
        # ################## 在这里设置类别信息
        # for i in range(self.num_stages):
        #     self.bbox_head[i].num_classes = kwargs['num_classes']

        # TODO: May add a new function in baseroihead
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]

                for i in range(num_imgs):
                    results = results_list[i]
                    # rename rpn_results.bboxes to rpn_results.priors
                    results.priors = results.pop('bboxes')

                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self.bbox_loss(stage, x, sampling_results,
                                          support_feats, support_labels, **kwargs)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self.mask_loss(stage, x, sampling_results,
                                              batch_gt_instances)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas)
                    # Empty proposal
                    if results_list is None:
                        break
        return losses
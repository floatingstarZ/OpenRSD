# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np

from mmdet.datasets.transforms.loading import LoadAnnotations
from mmrotate.registry import TRANSFORMS
@TRANSFORMS.register_module()
class LoadAnnotationsWithEx(LoadAnnotations):

    def __init__(
            self,
            with_obj_embed=False,
            *args,
            **kwargs) -> None:
        self.with_obj_embed = with_obj_embed
        super(LoadAnnotationsWithEx, self).__init__(*args, **kwargs)

    def _load_obj_embeds(self, results):

        gt_obj_embeds = []
        for instance in results.get('instances', []):
            gt_obj_embeds.append(instance['obj_embed'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_obj_embeds'] = np.array(gt_obj_embeds, dtype=np.float32)

        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_obj_embed:
            self._load_obj_embeds(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_obj_embed={self.with_obj_embed}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotationsWithText(LoadAnnotations):

    def __init__(
            self,
            with_text=False,
            *args,
            **kwargs) -> None:
        self.with_text = with_text
        super(LoadAnnotationsWithText, self).__init__(*args, **kwargs)

    def _load_texts(self, results):

        gt_texts = []
        for instance in results.get('instances', []):
            gt_texts.append(instance['text'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_texts'] = np.array(gt_texts)

        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_text:
            self._load_texts(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_text={self.with_text}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadAnnotationsMM(LoadAnnotations):

    def __init__(
            self,
            *args,
            **kwargs) -> None:
        super(LoadAnnotationsMM, self).__init__(*args, **kwargs)

    def _load_extra(self, results):
        results['texts'] = np.array([instance['text'] for instance in results['instances']])
        results['text_embeds'] = np.array([instance['text_embed'] for instance in results['instances']])
        results['text_masks'] = np.array([instance['text_mask'] for instance in results['instances']])
        results['visual_embeds'] = np.array([instance['visual_embed'] for instance in results['instances']])
        results['visual_masks'] = np.array([instance['visual_mask'] for instance in results['instances']])
        return results

    def transform(self, results: dict) -> dict:
        self._load_bboxes(results)
        self._load_extra(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
from mmdet.datasets.transforms.loading import get_box_type
import torch
@TRANSFORMS.register_module()
class LoadAnnotationsOnline(LoadAnnotations):

    def __init__(
            self,
            *args,
            **kwargs) -> None:
        super(LoadAnnotationsOnline, self).__init__(*args, **kwargs)

    def transform(self, results: dict) -> dict:
        results['texts'] = np.array(results['instances']['texts'])
        results['cls_list'] = np.array(results['instances']['cls_list'])
        results['ict_support_dict'] = results['instances']['ict_support_dict']
        results['text_embeds'] = np.array(results['instances']['text_embeds'])
        results['text_masks'] = np.array(results['instances']['text_masks'])
        results['visual_embeds'] = np.array(results['instances']['visual_embeds'])
        results['visual_masks'] = np.array(results['instances']['visual_masks'])
        if 'labels' in results['instances'].keys():
            results['gt_bboxes_labels'] = np.array(results['instances']['labels'], dtype=np.int64)

        gt_bboxes = results['instances']['boxes']
        gt_ignore_flags = results['instances']['ignore_flags']
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

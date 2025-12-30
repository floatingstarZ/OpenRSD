# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.evaluation.metrics import DOTAMetric

@METRICS.register_module()
class DETAILDOTAMetric(DOTAMetric):

    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            full_eval_results = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, full_eval_results_ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                #######################
                iou_str = str(int(iou_thr * 100))
                iou_eval_results = dict()
                for cls_name, eval_infos in zip(dataset_name, full_eval_results_):
                    if eval_infos['recall'].size > 0:
                        recall = round(float(eval_infos['recall'][-1]), 4)
                    else:
                        recall = 0.0
                    ap = round(float(eval_infos['ap']), 4)
                    num_dets = eval_infos['num_dets']
                    num_gts = eval_infos['num_gts']
                    cls_eval_results = dict()
                    cls_eval_results['ap'] = ap
                    cls_eval_results['recall'] = recall
                    cls_eval_results['num_dets'] = num_dets
                    cls_eval_results['num_gts'] = num_gts
                    iou_eval_results[cls_name] = cls_eval_results
                eval_results[f'IoU_{iou_str}_Detail'] = iou_eval_results

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results

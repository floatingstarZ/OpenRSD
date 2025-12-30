# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS
from pathlib import Path
from ctlib.os import *
import numpy as np
# from M_AD.datasets.dota_mm import DOTAMultiModalDataset
from mmengine.dataset.base_dataset import Compose
from copy import deepcopy
from mmdet.datasets.dataset_wrappers import ConcatDataset
from M_AD.datasets.dota_online_v1 import DOTADatasetOnline

@DATASETS.register_module()
class MultiViewDOTADatasetOnline(DOTADatasetOnline):
    def __init__(self,
                 *args,
                 view_pipelines=[],
                 **kwargs,):
        super(MultiViewDOTADatasetOnline, self).__init__(*args, **kwargs)
        # processing multi_views pipeline
        self.view_pipelines = []
        for pipe in view_pipelines:
            pipeline = Compose(pipe)
            self.view_pipelines.append(pipeline)

    def prepare_data(self, idx):
        rand_count = 0
        while True:
            data_info = self.get_data_info(idx)
            results = self.pipeline(deepcopy(data_info))
            ### ----------- PackDetInputsMM放在了最后，因此这里没有data_samples，还是results
            if len(results['gt_bboxes']) == 0 and self.filter_empty_gt:
                idx = self._rand_another()
                img_name = data_info['file_name']
                dataset_flag = data_info['dataset_flag']
                # print(f'Rand {rand_count}, Got no instance in {img_name}, in Data {dataset_flag}')
                rand_count += 1
            else:
                break
        out_results = list(map(lambda pipeline: pipeline(deepcopy(results)), self.view_pipelines))

        return out_results



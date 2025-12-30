# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sequence, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler
from mmdet.datasets.samplers import GroupMultiSourceSampler, MultiSourceSampler, AspectRatioBatchSampler


@DATA_SAMPLERS.register_module()
class MyMultiSourceSampler(MultiSourceSampler):
    """
    将MultiSourceSampler改为基于Epoch的Sampler
    原始的MultiSourceSampler是无限采样的，还是延续了以前的做法，只是对采样进行了Epoch的截断
    """

    def __init__(self,
                 max_iter_per_epoch,
                 batch_size: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_iter_per_epoch = max_iter_per_epoch
        self.dataset_size = self.max_iter_per_epoch * self.batch_size

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

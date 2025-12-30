# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sequence, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler
from mmdet.datasets.samplers import GroupMultiSourceSampler, MultiSourceSampler, AspectRatioBatchSampler


@DATA_SAMPLERS.register_module()
class MultiTaskSampler(MultiSourceSampler):
    """
    将MultiSourceSampler改为基于Epoch的Sampler
    原始的MultiSourceSampler是无限采样的，还是延续了以前的做法，只是对采样进行了Epoch的截断
    """

    def __init__(self, num_gpus, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cumulative_sizes = [0] + self.dataset.cumulative_sizes
        self.dataset_sizes = []
        for i in range(len(cumulative_sizes) - 1):
            self.dataset_sizes.append(cumulative_sizes[i+1] - cumulative_sizes[i])
        self.max_iters = []
        for num_sample, dataset_size in zip(self.num_per_source, self.dataset_sizes):
            self.max_iters.append(dataset_size // num_sample)

        max_iter = max(self.max_iters)
        # ---- 计算每个dataset能采样多少次，取最大的作为可迭代次数，然后反推一个epoch迭代多少次
        self.dataset_size = max_iter * self.batch_size // num_gpus
        self.max_iteration = max_iter // num_gpus
        print('#' * 100)
        print('$ MultiTaskSampler Initialized Infos: ')
        for i, size, sample, iter in zip(range(len(self.num_per_source)),
                                               self.dataset_sizes,
                                               self.num_per_source,
                                               self.max_iters):
            print(f'Dataset {i}: Size {size}; Sample: {sample}; Iter: {iter}')
        print(f'Set sampler len as {max_iter} * {self.batch_size} // {num_gpus} = {self.dataset_size}')
        print(f'Set Iteration as {max_iter} // {num_gpus} = {self.max_iteration}')
        print('#' * 100)

    def __iter__(self) -> Iterator[int]:
        indices = []
        for i in range(self.max_iteration):
            batch_buffer = []
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            indices.extend(batch_buffer)
        return iter(indices)

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sequence, Sized
import itertools
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler
from mmdet.datasets.samplers import GroupMultiSourceSampler, MultiSourceSampler, AspectRatioBatchSampler
import numpy as np
from tqdm import tqdm

@DATA_SAMPLERS.register_module()
class OneTaskSampler(Sampler):
    """
    将MultiSourceSampler改为基于Epoch的Sampler
    原始的MultiSourceSampler是无限采样的，还是延续了以前的做法，只是对采样进行了Epoch的截断
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 num_gpus=1,
                 source_prob=[],
                 max_iter_per_epoch: int = 1000
                 ) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }
        self.dataset_flags = [ds.dataset_flag for ds in dataset.datasets]

        cumulative_sizes = [0] + self.dataset.cumulative_sizes
        self.dataset_sizes = []
        for i in range(len(cumulative_sizes) - 1):
            self.dataset_sizes.append(cumulative_sizes[i+1] - cumulative_sizes[i])

        self.source_prob = source_prob
        probabilities = np.array(self.source_prob, dtype=np.float64)
        probabilities /= probabilities.sum()
        self.source_prob = probabilities
        assert len(self.source2inds) == len(source_prob)
        ##################################################
        ###  ------ 计算最大迭代次数（一个dataset轮转完）
        print('#' * 100)
        print('$ The source_probs have been normalized.')
        print('$ MultiTaskSampler Initialized Infos: ')
        mean_iters = []
        for i, dataset_size, sample_prob, flag in zip(range(len(self.source_prob)),
                                                self.dataset_sizes,
                                                self.source_prob,
                                                      self.dataset_flags):
            ### --- 每个数据集迭代次数 = 采样概率 * 总数据集大小 / （每批次采样个数 * GPU数量）
            ### --- 采样概率是该数据集在一轮中采样的个数
            prob = np.around(sample_prob, 3)
            if prob < 1e-6:
                mean_iter = 0
            else:
                mean_iter = int(dataset_size // (self.batch_size * num_gpus) / prob)
            print(f'Dataset {i}, {flag}: Size({dataset_size}) '
                  f'// (Batch_Size({self.batch_size}) * N_Gpu({num_gpus})) / Prob({prob}) = MeanIter({mean_iter})')
            mean_iters.append(mean_iter)
            if dataset_size == 0:
                raise Exception(f'Dataset {i} is Empty !!')
        max_mean_iters = max(mean_iters)
        self.max_iter_per_epoch = max_iter_per_epoch
        print(f'The maximum mean_iters: {max_mean_iters}')
        print(f'Manually set the maximum number of iterations per epoch to {self.max_iter_per_epoch}')

        # ----- task_sampler之后会再经过batch sampler进行采样，因此要乘上batch_size
        self.dataset_size = self.max_iter_per_epoch * batch_size
        print('#' * 100)

    def __iter__(self) -> Iterator[int]:
        indices = []
        print('#' * 100)
        print('Prepare sampling indices, in OneTaskSampler')
        for i in tqdm(list(range(self.max_iter_per_epoch))):
            batch_buffer = []

            # ---- 选择source
            source_ids = np.arange(0, len(self.source_prob))
            sampled_source_idx = int(np.random.choice(source_ids, 1, p=self.source_prob))

            # ---- source内采样
            count = 0
            for idx in self.source2inds[sampled_source_idx]:
                # print(idx)
                idx += self.cumulative_sizes[sampled_source_idx]
                batch_buffer.append(idx)
                count += 1
                if count == (self.batch_size * self.num_gpus):
                    break
            indices.extend(batch_buffer)
        return iter(indices)

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)


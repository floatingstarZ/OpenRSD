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
class TwoTaskSampler(Sampler):
    """
    将MultiSourceSampler改为基于Epoch的Sampler
    原始的MultiSourceSampler是无限采样的，还是延续了以前的做法，只是对采样进行了Epoch的截断
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 num_per_source: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 num_gpus=1,
                 source_prob=[],
                 ) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(num_per_source, list), \
            f'source_ratio must be a list, but got source_ratio={num_per_source}'
        assert len(num_per_source) == len(dataset.cumulative_sizes), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={num_per_source}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size

        self.num_per_source = num_per_source

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

        cumulative_sizes = [0] + self.dataset.cumulative_sizes
        self.dataset_sizes = []
        for i in range(len(cumulative_sizes) - 1):
            self.dataset_sizes.append(cumulative_sizes[i+1] - cumulative_sizes[i])

        self.source_prob = source_prob
        assert len(source_prob) == len(self.num_per_source)
        assert len(set(num_per_source)) == 1 # 只有一个元素
        ##################################################
        ###  ------ 计算最大迭代次数（一个dataset轮转完）
        print('#' * 100)
        print('$ MultiTaskSampler Initialized Infos: ')
        self.max_iters = []
        for i, num_sample, dataset_size, sample_prob in zip(range(len(num_per_source)),
                                                            self.num_per_source,
                                                            self.dataset_sizes,
                                                            self.source_prob):
            ### --- 每个数据集迭代次数 = 采样概率 * 总数据集大小 / （每批次采样个数 * GPU数量）
            ### --- 采样概率是该数据集在一轮中采样的个数
            max_iter = int(sample_prob * dataset_size // (num_sample * num_gpus))
            print(f'Dataset {i}: Prob({sample_prob}) * Size({dataset_size}) '
                  f'// (N_Sample({num_sample}) * N_Gpu({num_gpus})) = MaxIter({max_iter})')
            self.max_iters.append(max_iter)
            if dataset_size == 0:
                raise Exception(f'Dataset {i} is Empty !!')

        all_max_iter = max(self.max_iters)
        # ----- task_sampler之后会再经过batch sampler进行采样，因此要乘上batch_size
        self.dataset_size = all_max_iter * batch_size
        self.max_iter_per_epoch = all_max_iter
        print(f'Set Iteration as Max max_iter: {all_max_iter}')
        print('#' * 100)

    def __iter__(self) -> Iterator[int]:
        indices = []
        source_idx = 1
        num_per_source = self.num_per_source[0]
        print('#' * 100)
        print('Prepare sampling indices, in TwoTaskSampler')
        for i in tqdm(list(range(self.max_iter_per_epoch))):
            batch_buffer = []

            # ---- Basic Sampling (Pure Image)
            count = 0
            for idx in self.source2inds[0]:
                idx += self.cumulative_sizes[0]
                batch_buffer.append(idx)
                count += 1
                if count == num_per_source:
                    break
            # ---- Extra Sampling (Labeled Image)
            # ---- sampling source with prob
            for i in range(10):
                if source_idx >= len(self.num_per_source):
                    source_idx = 1
                prob = self.source_prob[source_idx]
                sample_prob = float(np.random.rand(1)[0])
                if np.random.rand(1)[0] >= prob:
                    # print(f'Continue in Source {source_idx}, '
                    #       f'with sample_prob {sample_prob} >= {prob}')
                    source_idx += 1
                    continue
                else:
                    # print(f'Break in Source {source_idx}, '
                    #       f'with sample_prob {sample_prob} < {prob}')
                    break

            count = 0
            for idx in self.source2inds[source_idx]:
                # print(idx)
                idx += self.cumulative_sizes[source_idx]
                batch_buffer.append(idx)
                count += 1
                if count == num_per_source:
                    break
            source_idx += 1
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


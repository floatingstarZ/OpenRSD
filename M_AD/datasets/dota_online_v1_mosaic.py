# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path
import os.path as osp
from typing import List, Tuple

import cv2

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS
from pathlib import Path
from ctlib.os import *
import numpy as np
import copy
import functools
import gc
import logging
import pickle
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from mmengine.config import Config
from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS
from mmengine.utils import is_abs
from mmengine.dataset.base_dataset import Compose, force_full_init
from copy import deepcopy
import itertools
import torch

@DATASETS.register_module()
class DOTADatasetOnlineMosaic(Dataset):

    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ###############
                 img_shape: Tuple[int, int] = (800, 800),
                 embed_dims=768,
                 dataset_flag='Skyscript',
                 ###############
                 img_dir = None,
                 ):
        self.ann_file = ann_file
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Join paths.
        self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        ##################
        self.img_shape = img_shape
        self.embed_dims = embed_dims
        ##################
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()
        self.dataset_flag = dataset_flag
        ##################
        self.img_dir = img_dir

    def full_init(self):
        ########## 初始化图像
        ann_files = glob.glob(osp.join(self.ann_file, '*.pkl'))
        self.ann_files = ann_files
        self.mosaic_sizes = [2, 3, 4]
        self.mosaic_ann_ids = self._inf_generator(len(self.ann_files))

        # for pkl_file in ann_files:
        #     pkl_file_pth = f'{self.ann_file}/{pkl_file}'
        #     if not os.path.exists(pkl_file_pth):
        #         print(f'Skip Empty {}'.format(pkl_file_pth))

        ##########
        if self._fully_initialized:
            return
        self._fully_initialized = True

    def _infinite_indices(self, sample_size: int):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(2024)
        while True:
            yield from torch.randperm(sample_size, generator=g).tolist()
            print('New_yield')

    def _inf_generator(self, sample_size):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), 0, None, 1)

    ###---------- 每次以idx为start，结合2*2、3*3、4*4个随机采样的其他样本，作为该次mosaic的输出
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data_infos = []
        base_data_info = self.get_data_info_single(idx)
        # ----- 采样剩余M * M - 1 张图片
        M = int(np.random.choice(np.array(self.mosaic_sizes)))

        count = 0
        data_infos = []
        data_infos.append(base_data_info)
        for m_idx in self.mosaic_ann_ids:
            if count >= M**2 - 1:
                break
            m_data_info = self.get_data_info_single(m_idx)
            data_infos.append(m_data_info)
            count += 1
        # ----- 图片拼接，构造Image Poly
        H, W = 512, 512
        a = int(W * 0.5  / 16)
        b = int(H * 15.5 / 16)
        polys = np.concatenate([np.array([[a, a, b, a, b, b, a, b]]) for i in range(len(data_infos))])
        out_H, out_W = self.img_shape
        texts = []
        for info in data_infos:
            texts.extend(info['instances']['texts'] )
        text_embeds = np.concatenate([info['instances']['text_embeds'] for info in data_infos])
        text_masks = np.concatenate([info['instances']['text_masks'] for info in data_infos])
        visual_embeds = np.concatenate([info['instances']['visual_embeds'] for info in data_infos])
        visual_masks = np.concatenate([info['instances']['visual_masks'] for info in data_infos])
        images = [cv2.resize(info['img'], (H, W)) for info in data_infos]

        spliced_img = np.zeros((H * M, W * M, 3), np.uint8)
        for i in range(M):
            for j in range(M):
                spliced_img[i * H: (i + 1) * H, j * W: (j + 1) * W] = images[i * M + j]
                polys[i * M + j, 0::2] = polys[i * M + j, 0::2] + j * W
                polys[i * M + j, 1::2] = polys[i * M + j, 1::2] + i * W

        s_H, s_W = spliced_img.shape[:2]
        polys[:, 0::2] = polys[:, 0::2] * out_H / s_H
        polys[:, 1::2] = polys[:, 1::2] * out_W / s_W

        spliced_img = cv2.resize(spliced_img, (out_H, out_W))
        ###### ----- 构造输出
        data_info = {}
        data_info['dataset_flag'] = base_data_info['dataset_flag']
        data_info['img_id'] = base_data_info['dataset_flag']
        data_info['file_name'] = base_data_info['file_name']
        data_info['img_path'] = base_data_info['img_path']
        data_info['img'] = spliced_img
        data_info['img_shape'] = spliced_img.shape[:2]
        data_info['ori_shape'] = spliced_img.shape[:2]

        data_info['height'] = self.img_shape[0]
        data_info['width'] = self.img_shape[1]

        instances = dict(
            texts=texts,
            text_embeds=text_embeds,
            text_masks=text_masks,
            visual_embeds=visual_embeds,
            visual_masks=visual_masks,
            ignore_flags=np.zeros(len(polys), dtype=np.uint8),
            boxes=polys,
            cls_list=list(set(texts))
        )

        data_info['instances'] = instances
        data_info['sample_idx'] = base_data_info['sample_idx']

        return data_info

    @force_full_init
    def get_data_info_single(self, idx: int) -> dict:
        pkl_file = copy.deepcopy(self.ann_files[idx])

        data_info = {}
        data_info['dataset_flag'] = self.dataset_flag
        img_id = osp.split(pkl_file)[1][:-4]
        data_info['img_id'] = img_id
        img_name = img_id + '.png'
        data_info['file_name'] = img_name
        data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                         img_name)
        img = cv2.imread(data_info['img_path'])
        data_info['img'] = img
        data_info['img_shape'] = img.shape[:2]
        data_info['ori_shape'] = img.shape[:2]


        data_info['height'] = self.img_shape[0]
        data_info['width'] = self.img_shape[1]
        ###########
        ann = pklload(pkl_file, msg=False)
        texts = ann['texts']
        n_instances = len(texts)

        text_embeds = ann['texts_embeds'] if ann['texts_embeds'] is not None \
            else np.zeros([n_instances, 768])
        text_masks = np.ones(n_instances, dtype=np.uint8) if ann['texts_embeds'] is not None \
            else np.zeros(n_instances, dtype=np.uint8)
        visual_embeds = ann['visual_embeds'] if ann['visual_embeds'] is not None \
            else np.zeros([n_instances, 1024])
        visual_masks = np.ones(n_instances, dtype=np.uint8) if ann['visual_embeds'] is not None \
            else np.zeros(n_instances, dtype=np.uint8)

        instances = dict(
            texts=texts,
            text_embeds=text_embeds,
            text_masks=text_masks,
            visual_embeds=visual_embeds,
            visual_masks=visual_masks,
            ignore_flags=np.zeros(len(texts), dtype=np.uint8),
            boxes=None # np.array(polys, dtype=np.float32)
        )

        data_info['instances'] = instances

        #########################
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def prepare_data(self, idx) -> Any:
        rand_count = 0
        while True:
            data_info = self.get_data_info(idx)
            out_data_info = self.pipeline(deepcopy(data_info))
            if len(out_data_info['data_samples'].gt_instances) == 0:
                idx = self._rand_another()
                img_name = data_info['file_name']
                dataset_flag = data_info['dataset_flag']
                print(f'Rand {rand_count}, Got no instance in {img_name}, in Data {dataset_flag}')
                rand_count += 1
            else:
                break

        return out_data_info

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
    @force_full_init
    def __len__(self) -> int:
        return len(self.ann_files)

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix,
                                                  raw_data_info[prefix_key])
        return raw_data_info

    def filter_data(self) -> List[dict]:
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  'method')

    @classmethod
    def _load_metainfo(cls,
                       metainfo: Union[Mapping, Config, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def _copy_without_annotation(self, memo=dict()) -> 'BaseDataset':
        """Deepcopy for all attributes other than ``data_list``,
        ``data_address`` and ``data_bytes``.

        Args:
            memo: Memory dict which used to reconstruct complex object
                correctly.
        """
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_bytes']:
                continue
            super(BaseDataset, other).__setattr__(key,
                                                  copy.deepcopy(value, memo))

        return other

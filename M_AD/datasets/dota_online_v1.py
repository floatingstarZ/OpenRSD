# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path
import os.path as osp
from typing import List, Tuple

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

@DATASETS.register_module()
class DOTADatasetOnline(Dataset):

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
                 img_shape: Tuple[int, int] = (1024, 1024),
                 embed_dims=768,
                 dataset_flag='Any',
                 max_instance=200000,
                 ###############
                 auto_generate_labels=False,
                 normalized_class_dict=None
                 ):
        self.ann_file = ann_file
        self.max_instance = max_instance

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

        if 'filter_empty_gt' in self.filter_cfg.keys():
            self.filter_empty_gt = self.filter_cfg['filter_empty_gt']
        else:
            self.filter_empty_gt = True

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

        self.auto_generate_labels = auto_generate_labels
        if self.auto_generate_labels:
            assert normalized_class_dict is not None
            self.norm_cls = pklload(normalized_class_dict)
            classes = [self.norm_cls[c] for c in self.metainfo['classes']]
            assert len(set(classes)) == len(self.metainfo['classes'])
            self.name2id = {c: i for i, c in enumerate(classes)}
            self.id2name = {i: c for i, c in enumerate(classes)}
        else:
            self.name2id = None
            self.id2name = None

    def full_init(self):
        ########## 初始化图像
        ann_files = glob.glob(osp.join(self.ann_file, '*.pkl'))
        self.ann_files = ann_files


        ################ DOTA2.0的>2000小目标实例测试
        # high_ins_ann_files = []
        # for ann_file in ann_files:
        #     ann_name = Path(ann_file).stem
        #     if ann_name in ['P10339__2048__2048___1048',
        #                         'P10339__2048__2048___2048',
        #                         'P1414__2048__2096___0',
        #                         'P1868__1024__5764___524',
        #                         'P1868__1024__6288___0',
        #                         'P1868__2048__0___1048',
        #                         'P1868__2048__5240___0',
        #                         'P1868__2048__6288___0',
        #                         'P4076__1024__524___1572',
        #                         'P4076__2048__0___2048',
        #                         'P4076__2048__1048___2048',
        #                         'P4076__2048__2049___0',
        #                         'P6525__2048__0___0',
        #                         'P6525__2048__0___1048',
        #                         'P7763__2048__0___2048',]:
        #         high_ins_ann_files.append(ann_file)
        # self.ann_files = high_ins_ann_files

        ##########
        if self._fully_initialized:
            return
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        #########################
        pkl_file = copy.deepcopy(self.ann_files[idx])

        data_info = {}
        data_info['dataset_flag'] = self.dataset_flag
        img_id = osp.split(pkl_file)[1][:-4]
        data_info['img_id'] = img_id
        img_name = img_id + '.png'
        data_info['file_name'] = img_name
        data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                         img_name)
        data_info['height'] = self.img_shape[0]
        data_info['width'] = self.img_shape[1]
        ###########
        ann = pklload(pkl_file, msg=False)
        texts = ann['texts']
        polys = ann['polys']
        n_instances = len(texts)
        ###########
        if self.auto_generate_labels:
            labels = [self.name2id[self.norm_cls[cls_name]] for cls_name in texts]
            labels = np.array(labels)

        text_embeds = ann['text_embeds'] if ann['text_embeds'] is not None \
            else np.zeros([n_instances, 768])
        text_masks = np.ones(n_instances, dtype=np.uint8) if ann['text_embeds'] is not None \
            else np.zeros(n_instances, dtype=np.uint8)
        visual_embeds = ann['visual_embeds'] if ann['visual_embeds'] is not None \
            else np.zeros([n_instances, 1024])
        visual_masks = np.ones(n_instances, dtype=np.uint8) if ann['visual_embeds'] is not None \
            else np.zeros(n_instances, dtype=np.uint8)
        cls_list = ann['cls_list'] if 'cls_list' in ann.keys() else None
        ict_support_dict = ann['ict_support_dict'] if 'ict_support_dict' in ann.keys() else None

        if n_instances >= self.max_instance:
            print(f'Warning: too many instances in {pkl_file}, '
                  f'cut {n_instances} to {self.max_instance}')
            # ---- 随机采样
            s = np.random.permutation(np.arange(n_instances))[:self.max_instance]
            texts = np.array(texts)[s]

            text_embeds = text_embeds[s]
            visual_embeds = visual_embeds[s]
            text_masks = text_masks[s]
            visual_masks = visual_masks[s]

            polys = np.array(polys, dtype=np.float32)[s]

        instances = dict(
            texts=texts,
            cls_list=cls_list,
            ict_support_dict=ict_support_dict,
            text_embeds=text_embeds,
            text_masks=text_masks,
            visual_embeds=visual_embeds,
            visual_masks=visual_masks,
            ignore_flags=np.zeros(len(polys), dtype=np.uint8),
            boxes=np.array(polys, dtype=np.float32)
        )

        if self.auto_generate_labels:
            instances['labels'] = labels


        data_info['instances'] = instances

        #########################
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        if self.dataset_flag == 'D6_Xview':
            a = 0

        return data_info

    def prepare_data(self, idx) -> Any:
        rand_count = 0
        while True:
            data_info = self.get_data_info(idx)
            out_data_info = self.pipeline(deepcopy(data_info))
            if len(out_data_info['data_samples'].gt_instances) == 0 and self.filter_empty_gt:
                idx = self._rand_another()
                img_name = data_info['file_name']
                dataset_flag = data_info['dataset_flag']
                # print(f'Rand {rand_count}, Got no instance in {img_name}, in Data {dataset_flag}')
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

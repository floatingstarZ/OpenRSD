# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from pathlib import Path
import torch
from copy import deepcopy
import numpy as np
from mmrotate.utils import register_all_modules
from ctlib.os import *
from mmcv.ops import box_iou_rotated
from M_Tools.Base_Data_infos.train_data_cfgs import train_cfgs
from M_Tools.Base_Data_infos.parent_mapping import uni_cls2parent, uni_parent2cls
from tqdm import tqdm
from mmcv.ops.nms import nms_rotated
from ctlib.rbox import obb2poly

score_thr = 0.2
new_iou = 0.005
nms_iou = 0.5

hard_match_iou = 0.5
out_label_root = '/data/space2/huangziyue/Formatted_FederatedLabels'
os.chdir('/opt/data/nfs/huangziyue/Projects/MMRotate_AD')
mkdir(out_label_root)
# for base_data_name in train_cfgs.keys():
#     src_ann_dir = os.path.join(train_cfgs[base_data_name]['data_root'],
#                                train_cfgs[base_data_name]['pkl_ann_dir'])
#     out_ann_dir = os.path.join(out_label_root, base_data_name)
#     mkdir(out_ann_dir)
#     for ann_file in tqdm(os.listdir(src_ann_dir)):
#         ann_pth = os.path.join(src_ann_dir, ann_file)
#         out_ann_pth = os.path.join(out_ann_dir, ann_file)
#         ann = pklload(ann_pth, msg=False)
#         a = 0
#         cls_list = deepcopy(train_cfgs[base_data_name]['class_names'])
#         for text in ann['texts']:
#             if text not in cls_list:
#                 raise Exception(f'{base_data_name}, Text {text} not in {cls_list}')
#         ann['cls_list'] = cls_list
#         pklsave(ann, out_ann_pth, msg=False)

######################
base_data_name = 'Data9_FMoW'
data_root = '/data/space2/huangziyue/DOTA2_1024_500/train'
src_ann_dir = f'{data_root}/Step6_Format_labels'
src_img_dir = f'{data_root}/images'
out_ann_dir = f'{data_root}/Step9_Make_Negative_Federate'
mkdir(out_ann_dir)
for img_file in tqdm(os.listdir(src_img_dir)):
    img_name = Path(img_file).stem
    ann_pth = f'{src_img_dir}/{img_file}.pkl'
    out_ann_pth = f'{out_ann_dir}/{img_file}.pkl'
    ann = pklload(ann_pth, msg=False)
    a = 0
    cls_list = deepcopy(['small-vehicle', 'storage-tank', 'large-vehicle',
           'plane', 'ship', 'harbor', 'tennis-court',
           'soccer-ball-field', 'swimming-pool', 'baseball-diamond',
           'ground-track-field', 'roundabout', 'basketball-court',
           'bridge', 'helicopter', 'airport', 'container-crane', 'helipad'])
    for text in ann['texts']:
        if text not in cls_list:
            raise Exception(f'{base_data_name}, Text {text} not in {cls_list}')
    ann['cls_list'] = cls_list
    pklsave(ann, out_ann_pth, msg=False)









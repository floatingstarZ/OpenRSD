# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

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

base_data_name = 'Data1_DOTA1'
src_ann_dir = os.path.join('./data/DOTA_800_600/train',
                           'labelTxt')
out_ann_dir = os.path.join(out_label_root, base_data_name)
mkdir(out_ann_dir)
for ann_file in tqdm(os.listdir(src_ann_dir)):
    ann_pth = os.path.join(src_ann_dir, ann_file)
    ann_name = Path(ann_pth).stem
    out_ann_pth = f'{out_ann_dir}/{ann_name}.pkl'

    ann_name = Path(ann_pth).stem
    with open(ann_pth) as f:
        lines = f.readlines()
        lines = [l.strip().split(' ') for l in lines]
    gt_polys = []
    gt_names = []
    for l in lines:
        poly = [float(coord) for coord in l[:8]]
        poly = np.array(poly)
        gt_polys.append(poly)
        gt_names.append(l[8])
    gt_polys = np.array(gt_polys)

    new_ann_data = dict(
        visual_embeds=None,
        texts=gt_names,
        text_embeds=None,
        polys=gt_polys,
        cls_list=['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                  'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                  'storage-tank', 'swimming-pool', 'tennis-court'],
    )

    pklsave(new_ann_data, out_ann_pth, msg=False)





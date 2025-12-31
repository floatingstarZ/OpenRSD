
import numpy as np

from commonlibs.common_tools import *
import os
import torch
import torch.nn.functional as F
from commonlibs.vis.detection import draw_dt_poly, draw_gt_poly
from pathlib import Path
import cv2
from copy import deepcopy
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.gmeans import gmeans
from ctlib.vis import draw_poly
import math
from tqdm import tqdm
from ctlib.rbox import *
from ctlib.dota import load_dota

data_root = '/data/space2/huangziyue/WHU_Mix/train'
ann_dir = f'{data_root}/labelTxt'
out_dir = f'{data_root}/Step6_Format_labels_wo_embeds'
mkdir(out_dir)
log_count = 0
for ann_file in tqdm(sorted(list(os.listdir(ann_dir)))):
    ann_pth = ann_dir + '/' + ann_file
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
        polys=gt_polys
    )
    out_ann_pth = f'{out_dir}/{ann_name}.pkl'
    pklsave(new_ann_data, out_ann_pth, msg=False)


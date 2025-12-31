
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

data_root = '/data/space2/huangziyue/DIOR_R_dota/train_val'
ann_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3'
out_dir = f'{data_root}/Step6_Format_labels'
mkdir(out_dir)
log_count = 0
for ann_file in tqdm(sorted(list(os.listdir(ann_dir)))):
    ann_data = pklload(ann_dir + '/' + ann_file, msg=False)
    embeds = ann_data['patch_feats']
    rboxes = ann_data['rboxes']
    texts = np.array(ann_data['cls_names'])

    in_class_ids = np.array(texts) != 'SAM_Obj'
    texts = texts[in_class_ids].tolist()
    rboxes = rboxes[in_class_ids]
    embeds = embeds[in_class_ids]
    if len(rboxes) < 0:
        print(f'Pass {ann_file} due to empty rboxes!')
        continue

    ######
    polys = obb2poly(rboxes)
    new_ann_data = dict(
        visual_embeds=embeds,
        texts=texts,
        text_embeds=None,
        polys=polys
    )
    out_ann_pth = f'{out_dir}/{ann_file}'
    pklsave(new_ann_data, out_ann_pth, msg=False)


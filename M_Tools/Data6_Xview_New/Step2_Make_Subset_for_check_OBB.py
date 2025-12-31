
import numpy as np

np.random.seed(42)

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

data_root = '/data/space2/huangziyue/xView_New_800_600/train'
img_dir = data_root + '/images'
ann_dir = data_root + '/Step1_Trans_HBB2OBB_IoU0.7'
out_root = data_root + '/Step2_Make_Subset_for_check_OBB_IoU07'
out_img_dir = out_root + '/images'
out_ann_dir = out_root + '/annotations'

mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)

Max_Sample = 200
count = 0

ann_files = sorted(list(os.listdir(ann_dir)))
ann_files = np.array(ann_files)
ann_files = np.random.permutation(ann_files)[:Max_Sample]
for ann_file in tqdm(ann_files):
    src_ann_pth = ann_dir + '/' + ann_file
    out_ann_pth = f'{out_ann_dir}/{ann_file}'
    os.system(f'cp {src_ann_pth} {out_ann_pth}')

    img_name = Path(ann_file).stem
    img_pth = f'{img_dir}/{img_name}.png'
    img = cv2.imread(img_pth)
    out_img_pth = f'{out_img_dir}/{img_name}.png'
    cv2.imwrite(out_img_pth, img)

    if count > Max_Sample:
        break
    count += 1

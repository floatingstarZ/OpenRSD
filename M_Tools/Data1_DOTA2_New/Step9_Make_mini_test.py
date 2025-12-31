
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
from ctlib.dota import load_dota

data_root = '/data/space2/huangziyue/DOTA2_1024_500/ss_val'
img_dir = data_root + '/images'
ann_dir = data_root + '/annfiles'
out_root = '/data/space2/huangziyue/MINI_Test_Dataset'
mkdir(out_root)
out_root = f'{out_root}/Data1_DOTA2'
out_img_dir = f'{out_root}/images'
out_ann_dir = f'{out_root}/annotations'

mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)

Max_Sample = 300
count = 0

ann_files = sorted(list(os.listdir(ann_dir)))
ann_files = np.array(ann_files)
ann_files = np.random.permutation(ann_files)[:Max_Sample]
for ann_file in tqdm(ann_files):
    src_ann_pth = f'{ann_dir}/{ann_file}'
    with open(src_ann_pth) as f:
        lines = f.readlines()
        lines = [l.strip().split(' ') for l in lines]
    gt_polys = []
    gt_names = []
    for l in lines:
        poly = [float(coord) for coord in l[:8]]
        poly = np.array(poly)
        gt_polys.append(poly)
        gt_names.append(l[8])
    if len(gt_polys) == 0:
        continue

    img_name = Path(ann_file).stem
    img_pth = f'{img_dir}/{img_name}.png'
    img = cv2.imread(img_pth)
    out_img_pth = f'{out_img_dir}/{img_name}.png'
    cv2.imwrite(out_img_pth, img)

    out_ann_pth = f'{out_ann_dir}/{ann_file}'
    os.system(f'cp {src_ann_pth} {out_ann_pth}')

    if count > Max_Sample:
        break
    count += 1

    #########################
    # ann_data = pklload(ann_dir + '/' + ann_file, msg=False)
    # out_ann_pth = f'{out_ann_dir}/{file_stem}.txt'
    # with open(out_ann_pth, 'wt+') as f:
    #     for cls_name, poly in zip(ann_data['texts'], ann_data['polys']):
    #         poly = poly.tolist()
    #         x1, y1, x2, y2, x3, y3, x4, y4 = poly
    #         difficulty = 0
    #         f.write('%.3f %.3f %.3f %.3f '
    #                 '%.3f %.3f %.3f %.3f %s %d\n' %
    #                 (x1, y1,
    #                  x2, y2,
    #                  x3, y3,
    #                  x4, y4,
    #                  cls_name, difficulty))
    ###########################
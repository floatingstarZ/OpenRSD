import json
import os
import pickle

import torch

from commonlibs.common_tools import *
from commonlibs.transform_tools.coco_annotation_template import COCOTmp
from commonlibs.transform_tools.data_transform import coco_transform
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from pathlib import Path
import cv2
from ctlib.rbox import *

log_dir = f'/data/space2/huangziyue/DOTA_800_600/train/7_18_Extract_Feats_DOTA_SAM_with_GT_DINOv2_ViTL'
pca_meta = pklload('/data/space2/huangziyue/DOTA_800_600/train/7_25_DIOR_DINOv2_tsne_pca/pca_meta.pkl')
out_label_dir = f'/data/space2/huangziyue/DOTA_800_600/train/7_25_DIOR_DINOv2_tsne_pca/cluster_labels'
mkdir(out_label_dir)

for log_file in tqdm(list(os.listdir(log_dir))):
    log_data = pklload(log_dir + '/' + log_file, msg=False)
    a = 0
    boxes = log_data['rois'][:, 1:]
    org_feats = log_data['patch_feats']
    img_meta = log_data['img_metas'][0]
    scale_factor = img_meta['scale_factor']
    scale_factor = np.array([scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]])

    file_name = Path(img_meta['img_path']).stem

    boxes[:, :4] = boxes[:, :4] / scale_factor[None, :]
    polys = obb2poly(boxes)

    out_ann_pth = out_label_dir + '/' + file_name + '.txt'
    A, b, ctrs = pca_meta['pca_A'], pca_meta['pca_b'], pca_meta['ctrs']

    pca_feats = org_feats @ A.T + b
    # row_sums = np.linalg.norm(pca_feats, axis=1)
    # pca_feats = pca_feats / row_sums[:, np.newaxis]
    cos_sims = pca_feats @ ctrs.T
    labels = np.argmax(cos_sims, axis=1)

    with open(out_ann_pth, 'wt+') as f:
        for poly, label in zip(polys, labels):
            poly = poly.clip(min=0)
            s = ''
            for p in poly.tolist():
                s += '%.1f ' % float(p)
            s += f'cluster_{int(label) + 1} 0\n'
            f.write(s)


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

src_ann_dir = f'/data/space2/huangziyue/DIOR_R_dota/train_val/7_21_Make_DIOR_pkl_anns_DINOv2'
pca_meta = pklload('/data/space2/huangziyue/DOTA_800_600/train/7_25_DIOR_DINOv2_tsne_pca/pca_meta.pkl')
out_label_dir = f'/data/space2/huangziyue/DIOR_R_dota/train_val/7_25_Make_DIOR_GT_pkl_anns_DINOv2_pca'
mkdir(out_label_dir)
A, b, ctrs = pca_meta['pca_A'], pca_meta['pca_b'], pca_meta['ctrs']

for ann_file in tqdm(list(os.listdir(src_ann_dir))):
    ann_data = pklload(src_ann_dir + '/' + ann_file, msg=False)
    org_feats = ann_data['visual_embeds']
    file_name = Path(ann_file).stem
    pca_feats = org_feats @ A.T + b

    out_ann_pth = out_label_dir + '/' + file_name + '.pkl'
    # row_sums = np.linalg.norm(pca_feats, axis=1)
    # pca_feats = pca_feats / row_sums[:, np.newaxis]
    cos_sims = pca_feats @ ctrs.T
    labels = np.argmax(cos_sims, axis=1)
    new_texts = [f'cluster_{int(label) + 1}' for label in labels]
    ann_data['texts'] = new_texts
    ann_data['visual_embeds'] = pca_feats
    pklsave(ann_data, out_ann_pth, msg=False)

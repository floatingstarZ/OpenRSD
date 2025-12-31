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

src_ann_dir = f'/data/space2/huangziyue/SkyScript_7_22_DINOv2_Mosaic_MiniTraing_S37_N10000/annotations'
pca_meta = pklload('/data/space2/huangziyue/DOTA_800_600/train/7_25_DIOR_DINOv2_tsne_pca/pca_meta.pkl')
out_label_dir = f'/data/space2/huangziyue/SkyScript_7_22_DINOv2_Mosaic_MiniTraing_S37_N10000/annotations_pca'
mkdir(out_label_dir)
A, b, ctrs = pca_meta['pca_A'], pca_meta['pca_b'], pca_meta['ctrs']

for ann_file in tqdm(list(os.listdir(src_ann_dir))):
    ann_data = pklload(src_ann_dir + '/' + ann_file, msg=False)
    org_feats = ann_data['visual_embeds']
    file_name = Path(ann_file).stem
    pca_feats = org_feats @ A.T + b
    ann_data['visual_embeds'] = pca_feats

    out_ann_pth = out_label_dir + '/' + file_name + '.pkl'
    pklsave(ann_data, out_ann_pth, msg=False)

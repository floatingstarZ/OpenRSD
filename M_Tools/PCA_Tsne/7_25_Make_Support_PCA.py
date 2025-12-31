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

pca_meta = pklload('/data/space2/huangziyue/DOTA_800_600/train/7_25_DIOR_DINOv2_tsne_pca/pca_meta.pkl')
out_support_pth = f'/data/space2/huangziyue/7_25_Make_Support_PCA.pkl'
A, b, ctrs = pca_meta['pca_A'], pca_meta['pca_b'], pca_meta['ctrs']
support_data = dict()
for i, ctr in enumerate(ctrs):
    support_data[f'cluster_{int(i) + 1}'] = dict(
        texts=[f'cluster_{int(i) + 1}', ],
        text_embeds=ctr.reshape(1, -1)
    )
pklsave(support_data, out_support_pth)
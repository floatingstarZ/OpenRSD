import os
import matplotlib as mpl
import cv2
# mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json
from ctlib.coco import COCOTmp

from ctlib.os import *
import os
import torch
import matplotlib.pyplot as plt
import random
from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
from sklearn.manifold import TSNE
from tqdm import tqdm
import colorsys
from pathlib import Path
from ctlib.os import *
from tqdm import tqdm
from pathlib import Path
import time

import numpy as np
import torch
import random
import torch.nn as nn
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
from ctlib.vis import plot_embedding_2d
################################### 文本部分 ###########################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
import torch, open_clip
from ctlib.vis import *
from ctlib.rbox import *
from ctlib.os import *
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import clip
from SkyScript_open_clip.factory import create_model_and_transforms, \
    create_model_from_pretrained, create_model
from ctlib.transform import to_array

##############################################################
data_root = '/data/space2/huangziyue/DOTA2_800_600/train'
src_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

out_data_root = '/data/space2/huangziyue/DOTA_800_600/train'
out_support_pth = f'{out_data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

dota2_class_names=['airport', 'baseball-diamond', 'basketball-court', 'bridge',
                     'container-crane', 'ground-track-field', 'harbor', 'helicopter', 'helipad',
                     'large-vehicle', 'plane', 'roundabout',
                     'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court']
dota1_class_names=['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                     'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court']


src_support_data = pklload(src_support_pth)
out_support_data = dict()
for k, v in src_support_data.items():
    if k not in dota1_class_names:
        print(f'Pass: {k}')
        continue
    out_support_data[k] = v
print(len(out_support_data))
pklsave(out_support_data, out_support_pth)


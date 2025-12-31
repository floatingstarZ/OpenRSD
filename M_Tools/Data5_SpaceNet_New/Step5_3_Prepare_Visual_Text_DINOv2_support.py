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
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
from ctlib.vis import plot_embedding_2d

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


data_root = '/data/space2/huangziyue/Spacenet_Merge'
feat_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'
from M_Tools.LabelSpace.Label3_Generate_Text_Prompts import Data5_SpaceNet

"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 20 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['buildings']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
phrases = Data5_SpaceNet

ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()

row_lables = []
support_data = dict()
for class_name, texts in phrases.items():
    row_lables.extend([class_name,] * len(texts))
    text = tokenizer(texts)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.cuda()).detach().cpu()
    text_embeds = to_array(text_features)
    text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
    support_data[class_name] = dict(
        texts=texts,
        text_embeds=text_embeds)

CLASSES =['building',]

cls_name = CLASSES[0]

id2name = {i:c for i, c in enumerate(CLASSES)}
name2id = {c:i for i, c in enumerate(CLASSES)}
# ----------------- 读取pkl数据，按照类别划分 -------------------
all_sample_feats = []
for log_file in tqdm(sorted(list(os.listdir(feat_dir)))):
    log_data = pklload(feat_dir + '/' + log_file, msg=False)
    cls_names = log_data['cls_names']
    gt_feats = log_data['patch_feats']
    gt_rboxes = log_data['rboxes']
    areas = gt_rboxes[:, 2] * gt_rboxes[:, 3]
    sort_idx = np.argsort(-areas)
    sampled_gt_feat = gt_feats[sort_idx][:2]
    all_sample_feats.append(sampled_gt_feat)
all_sample_feats = np.concatenate(all_sample_feats)
all_sample_feats = np.random.permutation(all_sample_feats)[:500]
support_data[cls_name]['visual_embeds'] = all_sample_feats
pklsave(support_data, out_support_pth)

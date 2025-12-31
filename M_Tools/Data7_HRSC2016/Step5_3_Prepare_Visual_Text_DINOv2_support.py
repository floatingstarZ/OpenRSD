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
data_root =  '/data/space2/huangziyue/HRSC2016_DOTA/train'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data7_HRSC2016_DINOv2_VitL_gt_MLP/epoch_100.pth')
###############################################################

ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()
"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 3 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['Arleigh_Burke', 'Austen',
           'Car_carrier', 'CntShip', 'Container', 'Cruise',
           'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
           'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
           'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
           'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['Arleigh_Burke', 'Austen',
           'Car_carrier', 'CntShip', 'Container', 'Cruise',
           'Enterprise', 'Hovercraft', 'Kuznetsov', 'Medical',
           'Midway_class', 'Nimitz', 'OXo', 'Perry', 'Sanantonio',
           'Tarawa', 'Ticonderoga', 'WhidbeyIsland', 'aircraft_carrier',
           'lute', 'merchant_ship', 'ship', 'submarine', 'warcraft', 'yacht']

phrases = {
    'Arleigh_Burke': [
        'An Arleigh Burke-class destroyer is visible in the satellite image.',
        'The aerial view shows an Arleigh Burke-class ship navigating through the sea.',
        'In the remote sensing image, an Arleigh Burke-class destroyer is clearly identified.'
    ],
    'Austen': [
        'An Austen-class amphibious transport dock is shown in the aerial photo.',
        'The remote sensing image captures an Austen-class ship near the coast.',
        'In the satellite image, an Austen-class vessel is prominently displayed.'
    ],
    'Car_carrier': [
        'A car carrier ship is visible in the remote sensing image, loaded with vehicles.',
        'The aerial photograph reveals a car carrier navigating the harbor.',
        'In the satellite image, a large car carrier is seen transporting cars across the water.'
    ],
    'CntShip': [
        'A container ship is detected in the aerial view, with its cargo clearly visible.',
        'The remote sensing image shows a large container ship docked at the port.',
        'In the satellite photo, a container ship can be seen amidst the bustling harbor.'
    ],
    'Container': [
        'A container ship appears in the remote sensing image, filled with stacked containers.',
        'The aerial view reveals a container vessel maneuvering through the shipping lanes.',
        'In the satellite image, a ship carrying numerous containers is identifiable.'
    ],
    'Cruise': [
        'A luxury cruise ship is visible in the aerial image, cruising through calm waters.',
        'The remote sensing image shows a large cruise ship docked at a popular tourist port.',
        'In the satellite photo, a cruise liner is seen navigating the open sea.'
    ],
    'Enterprise': [
        'The remote sensing image features the USS Enterprise, an iconic aircraft carrier.',
        'An aerial photograph captures the USS Enterprise in a naval exercise.',
        'In the satellite view, the USS Enterprise is prominently displayed in the harbor.'
    ],
    'Hovercraft': [
        'A hovercraft is seen in the aerial image, gliding over the water surface.',
        'The remote sensing image shows a hovercraft navigating in shallow waters.',
        'In the satellite photo, a hovercraft is visible approaching the shore.'
    ],
    'Kuznetsov': [
        'The Russian aircraft carrier Kuznetsov is visible in the remote sensing image.',
        'An aerial view shows the Kuznetsov carrier conducting operations at sea.',
        'In the satellite photo, the Kuznetsov aircraft carrier is clearly identifiable.'
    ],
    'Medical': [
        'A medical evacuation ship is seen in the remote sensing image, stationed offshore.',
        'The aerial photograph reveals a medical ship providing assistance near the coast.',
        'In the satellite image, a medical vessel is visible in a disaster relief operation.'
    ],
    'Midway_class': [
        'A Midway-class aircraft carrier is captured in the remote sensing image, moored at port.',
        'The aerial view shows a Midway-class ship in the middle of a naval fleet.',
        'In the satellite photo, a Midway-class aircraft carrier is clearly distinguished.'
    ],
    'Nimitz': [
        'The Nimitz-class aircraft carrier is visible in the remote sensing image, sailing across the ocean.',
        'An aerial view reveals the Nimitz-class carrier conducting maneuvers at sea.',
        'In the satellite photo, the Nimitz-class aircraft carrier is prominently displayed.'
    ],
    'OXo': [
        'An OXO-class naval vessel is seen in the remote sensing image, docked at a naval base.',
        'The aerial photograph shows an OXO-class ship in a busy harbor.',
        'In the satellite image, an OXO-class vessel is identifiable in the water.'
    ],
    'Perry': [
        'A Perry-class frigate is visible in the aerial image, patrolling the waters.',
        'The remote sensing image captures a Perry-class ship in a naval exercise.',
        'In the satellite view, a Perry-class frigate is clearly seen near the coastline.'
    ],
    'Sanantonio': [
        'A San Antonio-class amphibious transport dock is detected in the remote sensing image.',
        'The aerial photograph shows a San Antonio-class ship docking at a military base.',
        'In the satellite photo, a San Antonio-class vessel is visible in the harbor.'
    ],
    'Tarawa': [
        'The Tarawa-class amphibious assault ship is visible in the remote sensing image, anchored offshore.',
        'An aerial view shows a Tarawa-class ship conducting operations near a coastal area.',
        'In the satellite image, a Tarawa-class vessel is seen in the marine environment.'
    ],
    'Ticonderoga': [
        'A Ticonderoga-class cruiser is visible in the remote sensing image, cruising through open waters.',
        'The aerial photograph shows a Ticonderoga-class ship in a naval formation.',
        'In the satellite view, a Ticonderoga-class cruiser is identifiable in the fleet.'
    ],
    'WhidbeyIsland': [
        'A Whidbey Island-class dock landing ship is seen in the aerial image, preparing for amphibious operations.',
        'The remote sensing image captures a Whidbey Island-class vessel at sea.',
        'In the satellite photo, a Whidbey Island-class ship is visible near the shore.'
    ],
    'aircraft_carrier': [
        'An aircraft carrier is visible in the remote sensing image, moving through the sea.',
        'The aerial photograph reveals an aircraft carrier docked at a naval base.',
        'In the satellite image, an aircraft carrier is clearly identified in the water.'
    ],
    'lute': [
        'A lute-class naval vessel is seen in the remote sensing image, positioned in a strategic location.',
        'The aerial view shows a lute-class ship maneuvering through the harbor.',
        'In the satellite photo, a lute-class vessel is visible at sea.'
    ],
    'merchant_ship': [
        'A merchant ship is visible in the remote sensing image, transporting cargo across the ocean.',
        'The aerial photograph captures a merchant vessel docked at a busy port.',
        'In the satellite view, a merchant ship is clearly identified in the shipping lane.'
    ],
    'ship': [
        'A ship is detected in the remote sensing image, sailing through open waters.',
        'The aerial view shows a ship approaching the harbor.',
        'In the satellite photo, a ship is visible amidst the marine landscape.'
    ],
    'submarine': [
        'A submarine is visible in the remote sensing image, submerged near the surface.',
        'The aerial photograph reveals a submarine surfaced during a naval exercise.',
        'In the satellite view, a submarine is identified beneath the water surface.'
    ],
    'warcraft': [
        'A warcraft is seen in the remote sensing image, participating in a naval maneuver.',
        'The aerial view shows a warcraft docked at a military port.',
        'In the satellite photo, a warcraft is clearly visible amidst other naval vessels.'
    ],
    'yacht': [
        'A luxury yacht is visible in the remote sensing image, cruising along the coastline.',
        'The aerial photograph captures a yacht docked in a scenic marina.',
        'In the satellite image, a yacht is seen sailing through tranquil waters.'
    ]
}




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

################################### 视觉部分 ###########################
class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.fc(x)
class cls_model(torch.nn.Module):
    def __init__(self):
        super(cls_model, self).__init__()
        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.head = head()
    def forward(self, x):
        x = self.neck(x)
        cls_score = self.head(x)
        return cls_score


num_classes = len(classes)

classifier = cls_model()
classifier.load_state_dict(torch.load(classification_ckpt_pth)['state_dict'])
classifier.eval()

max_support_shot = 50
min_score_thr = 0.5
# 好像是按照排序后的CLASSES来获得ID的
for cls_id, cls_name in enumerate(sorted(classes)):
    cls_feats = []
    cls_dir = f'{val_gt_dir}/{cls_name}'
    for feat_file in os.listdir(f'{val_gt_dir}/{cls_name}'):
        feat_pth = f'{cls_dir}/{feat_file}'
        feat = pklload(feat_pth, msg=False)
        cls_feats.append(feat)

    #### ---- Add train if not enough feats
    if len(cls_feats) < 50:
        cls_dir = f'{train_gt_dir}/{cls_name}'
        for feat_file in os.listdir(f'{train_gt_dir}/{cls_name}'):
            feat_pth = f'{cls_dir}/{feat_file}'
            feat = pklload(feat_pth, msg=False)
            cls_feats.append(feat)
    cls_feats = np.stack(cls_feats)
    cls_feats = torch.Tensor(cls_feats)
    cls_scores = classifier(cls_feats).softmax(dim=1)[:, cls_id]
    sort_idx = torch.argsort(-cls_scores)

    selected_idx = sort_idx[:max_support_shot]
    confidence = cls_scores[selected_idx]
    embeds = cls_feats[selected_idx]

    pos_idx = confidence >= min_score_thr
    if torch.sum(pos_idx) != 0:
        confidence = confidence[pos_idx]
        embeds = embeds[pos_idx]
    ######## Padding
    if len(embeds) != max_support_shot:
        embeds_pad = torch.cat([embeds for i in range(max_support_shot)])
        embeds = embeds_pad[:max_support_shot]
        confidence_pad = torch.cat([confidence for i in range(max_support_shot)])
        confidence = confidence_pad[:max_support_shot]

    if cls_name not in support_data.keys():
        raise Exception(f'{cls_name} not in visual embeds')
    support_data[cls_name]['visual_embeds'] = embeds.detach().cpu().numpy()
    support_data[cls_name]['confidence_scores'] = confidence.detach().cpu().numpy()
pklsave(support_data, out_support_pth)



############################# 可视化分析 ##################################

all_text_features = np.concatenate([info['text_embeds'] for cls_name, info in support_data.items()])
text_sims = np.matmul(all_text_features, all_text_features.T)
text_sims = np.around(text_sims, 2)
print(text_sims)
print('Min', np.min(text_sims))
print('Max-Min', np.max(text_sims) - np.min(text_sims))
fig, ax = plt.subplots(figsize=(128,128))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.set(font_scale=1.25)
hm = sns.heatmap(text_sims,
                 cbar=False,
                 annot=True, # 注入数字
                 # square=True, # 单元格为正方形
                 fmt='.2f',   # 字符串格式代码
                  annot_kws={'size': 10}, # 当annot为True时，ax.text的关键字参数，即注入数字的字体大小
                  yticklabels=row_lables,  # 列标签
                  xticklabels=row_lables   # 行标签
                  )
plt.savefig(out_fig_pth)
plt.close()








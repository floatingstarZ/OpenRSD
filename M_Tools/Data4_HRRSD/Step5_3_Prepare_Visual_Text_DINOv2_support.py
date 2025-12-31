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
data_root = '/data/space2/huangziyue/HRRSD_800_0/train'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data4_HRRSD_DINOv2_VitL_gt_MLP/epoch_100.pth')
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
Each category generates 7 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['airplane', 'storage', 'bridge', 'ground', 'basketball',
          'tennis', 'ship', 'baseball',
          'T', 'crossroad', 'parking', 'harbor', 'vehicle']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['airplane', 'storage', 'bridge', 'ground', 'basketball',
          'tennis', 'ship', 'baseball',
          'T', 'crossroad', 'parking', 'harbor', 'vehicle']
phrases = {
    'airplane': [
        'An airplane is visible in the remote sensing image.',
        'The aerial image shows an airplane on the runway.',
        'A jet appears in the satellite photo.',
        'An airplane flying over the landscape is captured in this image.',
        'The image reveals an airplane at an airport.',
        'A plane is seen soaring in the clear sky.',
        'An aircraft is depicted in the overhead shot.'
    ],
    'storage': [
        'Storage units are present in the aerial view.',
        'The image captures storage facilities in an industrial area.',
        'Aerial imagery shows a series of storage warehouses.',
        'Storage buildings are visible in the remote sensing image.',
        'An overhead shot reveals multiple storage structures.',
        'The photo highlights storage units amidst other buildings.',
        'Storage areas can be seen in the satellite image.'
    ],
    'bridge': [
        'A bridge spans across the river in the aerial image.',
        'The remote sensing image captures a bridge over the water.',
        'A bridge is visible connecting two landmasses.',
        'The image depicts a bridge with vehicles crossing it.',
        'An overhead view shows a long bridge.',
        'A bridge is captured in the image, arching over a canyon.',
        'The photo reveals a bridge amid a bustling cityscape.'
    ],
    'ground': [
        'Ground features are prominent in the satellite image.',
        'The aerial view highlights the ground terrain.',
        'An image shows various ground textures and surfaces.',
        'The ground is visible with different land uses in the photo.',
        'A remote sensing image displays diverse ground patterns.',
        'The ground is clearly seen in the rural area image.',
        'Ground details are captured in the urban landscape.'
    ],
    'basketball': [
        'A basketball court is visible in the aerial image.',
        'The photo shows a basketball game in progress on the court.',
        'An outdoor basketball court is depicted in the satellite image.',
        'The image captures a basketball court in a park.',
        'A basketball court is seen amidst a residential area.',
        'The aerial view shows people playing basketball.',
        'A basketball court is clearly marked in the school yard image.'
    ],
    'tennis': [
        'A tennis court is visible in the remote sensing image.',
        'The aerial view captures a tennis match in progress.',
        'An image shows multiple tennis courts in a sports complex.',
        'The satellite photo highlights a tennis court in a park.',
        'A tennis court is seen surrounded by greenery.',
        'The photo reveals players on a tennis court.',
        'An overhead view shows a tennis court with clear markings.'
    ],
    'ship': [
        'A ship is seen in the harbor in the aerial image.',
        'The remote sensing image shows a ship sailing in the ocean.',
        'An image captures a large ship docked at the port.',
        'The photo depicts a ship navigating through the sea.',
        'A ship is visible near the coastline in the satellite image.',
        'The aerial view reveals a cargo ship at sea.',
        'A ship is shown anchored in the bay.'
    ],
    'baseball': [
        'A baseball field is visible in the aerial image.',
        'The remote sensing image captures a baseball game in progress.',
        'An image shows a baseball diamond in a park.',
        'The photo depicts a baseball field with players on it.',
        'A baseball field is seen surrounded by bleachers.',
        'The aerial view shows a baseball field in a school yard.',
        'A baseball game is visible in the stadium image.'
    ],
    'T': [
        'A T-junction is visible in the aerial image.',
        'The remote sensing image captures a T-shaped intersection.',
        'An image shows a T-junction in an urban area.',
        'The photo depicts a T-intersection with traffic.',
        'A T-junction is seen amidst residential streets.',
        'The aerial view shows a T-intersection in a rural area.',
        'A T-shaped road junction is clearly visible in the image.'
    ],
    'crossroad': [
        'A crossroad is visible in the aerial image.',
        'The remote sensing image captures a busy intersection.',
        'An image shows a crossroad with multiple lanes.',
        'The photo depicts a crossroad in a bustling city.',
        'A crossroad is seen connecting four streets.',
        'The aerial view shows a crossroad with traffic signals.',
        'A major crossroad is clearly visible in the urban image.'
    ],
    'parking': [
        'A parking lot is visible in the aerial image.',
        'The remote sensing image captures a full parking area.',
        'An image shows a parking lot with many vehicles.',
        'The photo depicts a parking area next to a shopping mall.',
        'A parking lot is seen in the satellite image.',
        'The aerial view shows a parking lot in a busy area.',
        'A large parking area is visible in the industrial zone image.'
    ],
    'harbor': [
        'A harbor is visible in the aerial image.',
        'The remote sensing image captures ships docked in a harbor.',
        'An image shows a harbor with various boats and ships.',
        'The photo depicts a busy harbor with cargo ships.',
        'A harbor is seen in the coastal area image.',
        'The aerial view shows a harbor with fishing boats.',
        'A large harbor is clearly visible in the image.'
    ],
    'vehicle': [
        'Vehicles are visible in the aerial image.',
        'The remote sensing image captures cars on the highway.',
        'An image shows vehicles parked along the streets.',
        'The photo depicts vehicles moving through a city.',
        'Vehicles are seen in the image of a traffic intersection.',
        'The aerial view shows a row of parked vehicles.',
        'Various types of vehicles are visible in the urban image.'
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








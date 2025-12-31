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
data_root = '/data/space2/huangziyue/DIOR_R_dota/train_val'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data2_DIOR_R_DINOv2_VitL_gt_MLP/epoch_100.pth')
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
Each category generates 10 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['golffield', 'vehicle', 'Expressway-toll-station',
           'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
           'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
           'basketballcourt', 'Expressway-Service-area', 'stadium',
           'airport', 'baseballfield', 'bridge', 'windmill', 'overpass']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""

classes = ['golffield', 'vehicle', 'Expressway-toll-station',
           'trainstation', 'chimney', 'storagetank', 'ship', 'harbor',
           'airplane', 'tenniscourt', 'groundtrackfield', 'dam',
           'basketballcourt', 'Expressway-Service-area', 'stadium',
           'airport', 'baseballfield', 'bridge', 'windmill', 'overpass']
phrases = {
    'golffield': [
        'A golf field visible in the satellite image.',
        'The aerial photo shows a sprawling golf field.',
        'Golf fields appear in the overhead shot.',
        'An expanse of golf field captured in the drone image.',
        'A lush golf field in the remote sensing image.',
        'Golf fields under a clear sky in the aerial image.',
        'A golf field surrounded by trees in the satellite picture.',
        'Aerial imagery captures the golf field.',
        'The remote sensing photo includes a golf field.',
        'A golf field during a sunny day in the satellite image.'
    ],
    'vehicle': [
        'Several vehicles seen in the aerial image.',
        'A car moving along the road in the drone image.',
        'A remote sensing image shows multiple vehicles.',
        'Vehicles scattered across the parking lot in the satellite image.',
        'A vehicle traveling in the overhead shot.',
        'The aerial photo captures a line of vehicles.',
        'Vehicles visible on the highway in the satellite image.',
        'A cluster of vehicles in the drone picture.',
        'A vehicle parked near the building in the aerial photo.',
        'Remote sensing captures a vehicle on the street.'
    ],
    'Expressway-toll-station': [
        'An expressway toll station seen in the satellite image.',
        'The aerial image shows an expressway toll station.',
        'A toll station on the expressway in the drone photo.',
        'Remote sensing captures the expressway toll station.',
        'An expressway toll station visible in the overhead shot.',
        'A line of cars at the toll station in the aerial image.',
        'The satellite picture includes an expressway toll station.',
        'A toll station on a busy expressway in the drone image.',
        'The expressway toll station seen in the remote sensing image.',
        'Aerial imagery shows an expressway toll station.'
    ],
    'trainstation': [
        'A train station visible in the aerial image.',
        'The drone photo shows a bustling train station.',
        'A remote sensing image captures the train station.',
        'A train station with several tracks in the satellite picture.',
        'The aerial shot includes a train station.',
        'A train station during rush hour in the overhead image.',
        'A train station surrounded by buildings in the drone picture.',
        'Remote sensing shows the train station clearly.',
        'The train station seen from above in the satellite image.',
        'A train station bustling with activity in the aerial photo.'
    ],
    'chimney': [
        'A chimney visible in the satellite image.',
        'The aerial image shows a tall chimney.',
        'A remote sensing image captures a smoking chimney.',
        'The drone picture includes an industrial chimney.',
        'A chimney releasing smoke in the overhead shot.',
        'The aerial photo captures a factory chimney.',
        'A chimney standing tall in the satellite picture.',
        'Remote sensing shows a chimney in an industrial area.',
        'A chimney with smoke in the aerial image.',
        'A tall chimney seen in the remote sensing image.'
    ],
    'storagetank': [
        'A storage tank seen in the aerial image.',
        'The drone photo shows a large storage tank.',
        'A remote sensing image captures several storage tanks.',
        'The satellite picture includes a storage tank.',
        'A storage tank in an industrial area in the overhead shot.',
        'The aerial image shows multiple storage tanks.',
        'A storage tank visible in the drone picture.',
        'Remote sensing captures a large storage tank.',
        'Aerial imagery shows a storage tank at a facility.',
        'A storage tank seen in the satellite image.'
    ],
    'ship': [
        'A ship visible in the aerial image.',
        'The drone photo captures a large ship.',
        'A remote sensing image shows a ship at sea.',
        'The satellite picture includes a ship near the harbor.',
        'A ship sailing in the overhead shot.',
        'A ship docked at the port in the aerial photo.',
        'The aerial image shows a cargo ship.',
        'Remote sensing captures a ship on the water.',
        'A ship seen from above in the satellite image.',
        'A large ship in the aerial image.'
    ],
    'harbor': [
        'A harbor visible in the satellite image.',
        'The aerial image shows a busy harbor.',
        'A remote sensing image captures the harbor.',
        'The drone picture includes a harbor with many ships.',
        'A harbor full of vessels in the overhead shot.',
        'A harbor seen in the aerial photo.',
        'The satellite image shows a bustling harbor.',
        'Remote sensing captures a harbor area.',
        'A harbor with boats in the aerial image.',
        'Aerial imagery shows a large harbor.'
    ],
    'airplane': [
        'An airplane visible in the aerial image.',
        'The drone photo captures an airplane on the runway.',
        'A remote sensing image shows an airplane in flight.',
        'The satellite picture includes an airplane at the airport.',
        'An airplane taking off in the overhead shot.',
        'An airplane on the tarmac in the aerial photo.',
        'The aerial image shows an airplane at the gate.',
        'Remote sensing captures an airplane in the sky.',
        'An airplane seen from above in the satellite image.',
        'An airplane parked in the aerial image.'
    ],
    'tenniscourt': [
        'A tennis court seen in the aerial image.',
        'The drone photo shows a tennis court.',
        'A remote sensing image captures several tennis courts.',
        'The satellite picture includes a tennis court.',
        'A tennis court in the overhead shot.',
        'The aerial image shows a tennis court in a park.',
        'A tennis court visible in the drone picture.',
        'Remote sensing captures a tennis court in use.',
        'Aerial imagery shows a tennis court with players.',
        'A tennis court seen in the satellite image.'
    ],
    'groundtrackfield': [
        'A ground track field visible in the satellite image.',
        'The aerial image shows a ground track field.',
        'A remote sensing image captures the ground track field.',
        'The drone picture includes a ground track field with runners.',
        'A ground track field in the overhead shot.',
        'The aerial photo captures a ground track field in use.',
        'A ground track field surrounded by spectators in the satellite picture.',
        'Remote sensing shows the ground track field clearly.',
        'A ground track field seen from above in the aerial image.',
        'A ground track field during a race in the satellite image.'
    ],
    'dam': [
        'A dam visible in the satellite image.',
        'The aerial image shows a large dam.',
        'A remote sensing image captures the dam.',
        'The drone picture includes a dam with water.',
        'A dam in the overhead shot.',
        'The aerial photo captures a dam in operation.',
        'A dam seen in the satellite picture.',
        'Remote sensing shows the dam clearly.',
        'A dam in the mountains in the aerial image.',
        'A large dam in the satellite image.'
    ],
    'basketballcourt': [
        'A basketball court seen in the aerial image.',
        'The drone photo shows a basketball court.',
        'A remote sensing image captures several basketball courts.',
        'The satellite picture includes a basketball court.',
        'A basketball court in the overhead shot.',
        'The aerial image shows a basketball court in a park.',
        'A basketball court visible in the drone picture.',
        'Remote sensing captures a basketball court in use.',
        'Aerial imagery shows a basketball court with players.',
        'A basketball court seen in the satellite image.'
    ],
    'Expressway-Service-area': [
        'An expressway service area seen in the aerial image.',
        'The drone photo shows an expressway service area.',
        'A remote sensing image captures the expressway service area.',
        'The satellite picture includes an expressway service area.',
        'An expressway service area in the overhead shot.',
        'The aerial image shows a busy expressway service area.',
        'An expressway service area visible in the drone picture.',
        'Remote sensing captures an expressway service area with facilities.',
        'Aerial imagery shows an expressway service area with vehicles.',
        'An expressway service area seen in the satellite image.'
    ],
    'stadium': [
        'A stadium visible in the satellite image.',
        'The aerial image shows a large stadium.',
        'A remote sensing image captures the stadium.',
        'The drone picture includes a stadium full of spectators.',
        'A stadium in the overhead shot.',
        'The aerial photo captures a stadium during an event.',
        'A stadium seen in the satellite picture.',
        'Remote sensing shows the stadium clearly.',
        'A stadium in the city in the aerial image.',
        'A large stadium in the satellite image.'
    ],
    'airport': [
        'An airport visible in the aerial image.',
        'The drone photo captures a busy airport.',
        'A remote sensing image shows an airport with multiple runways.',
        'The satellite picture includes an airport terminal.',
        'An airport seen in the overhead shot.',
        'The aerial image shows planes at the airport.',
        'An airport with many planes in the drone picture.',
        'Remote sensing captures an airport from above.',
        'An airport seen in the satellite image.',
        'A large airport in the aerial image.'
    ],
    'baseballfield': [
        'A baseball field seen in the aerial image.',
        'The drone photo shows a baseball field.',
        'A remote sensing image captures a baseball field with players.',
        'The satellite picture includes a baseball field.',
        'A baseball field in the overhead shot.',
        'The aerial image shows a baseball field in a park.',
        'A baseball field visible in the drone picture.',
        'Remote sensing captures a baseball field in use.',
        'Aerial imagery shows a baseball field with spectators.',
        'A baseball field seen in the satellite image.'
    ],
    'bridge': [
        'A bridge visible in the satellite image.',
        'The aerial image shows a long bridge.',
        'A remote sensing image captures the bridge over water.',
        'The drone picture includes a bridge with traffic.',
        'A bridge seen in the overhead shot.',
        'The aerial photo captures a bridge connecting two areas.',
        'A bridge seen in the satellite picture.',
        'Remote sensing shows the bridge clearly.',
        'A bridge crossing a river in the aerial image.',
        'A large bridge in the satellite image.'
    ],
    'windmill': [
        'A windmill seen in the aerial image.',
        'The drone photo shows a windmill farm.',
        'A remote sensing image captures several windmills.',
        'The satellite picture includes a windmill in a field.',
        'A windmill in the overhead shot.',
        'The aerial image shows a windmill generating power.',
        'A windmill visible in the drone picture.',
        'Remote sensing captures a windmill in operation.',
        'Aerial imagery shows a windmill on a hill.',
        'A windmill seen in the satellite image.'
    ],
    'overpass': [
        'An overpass visible in the aerial image.',
        'The drone photo shows a busy overpass.',
        'A remote sensing image captures an overpass with vehicles.',
        'The satellite picture includes an overpass connecting highways.',
        'An overpass seen in the overhead shot.',
        'The aerial image shows an overpass during rush hour.',
        'An overpass visible in the drone picture.',
        'Remote sensing captures an overpass in the city.',
        'Aerial imagery shows an overpass with traffic.',
        'An overpass seen in the satellite image.'
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








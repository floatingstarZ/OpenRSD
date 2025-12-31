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
data_root = '/data/space2/huangziyue/FAIR1M_1024_0/train'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data3_FAIR1M_DINOv2_VitL_gt_MLP/epoch_100.pth')
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

Categories Has: "classes = ['A220', 'A321', 'A330', 'A350', 'ARJ21',
           'Baseball', 'Basketball', 'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
           'Bridge', 'Bus', 'C919', 'Cargo', 'Dry', 'Dump', 'Engineering', 'Excavator',
           'Fishing', 'Football', 'Intersection', 'Liquid', 'Motorboat', 'Passenger',
           'Roundabout', 'Small', 'Tennis', 'Tractor', 'Trailer', 'Truck', 'Tugboat',
           'Van', 'Warship', 'other-airplane', 'other-ship', 'other-vehicle']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['A220', 'A321', 'A330', 'A350', 'ARJ21',
           'Baseball', 'Basketball', 'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
           'Bridge', 'Bus', 'C919', 'Cargo', 'Dry', 'Dump', 'Engineering', 'Excavator',
           'Fishing', 'Football', 'Intersection', 'Liquid', 'Motorboat', 'Passenger',
           'Roundabout', 'Small', 'Tennis', 'Tractor', 'Trailer', 'Truck', 'Tugboat',
           'Van', 'Warship', 'other-airplane', 'other-ship', 'other-vehicle']
phrases = {
    'A220': [
        "An A220 jet is visible in the aerial photograph.",
        "The image captures an A220 aircraft flying over the city.",
        "A clear view of an A220 in the remote sensing image."
    ],
    'A321': [
        "An aerial shot featuring the A321 on the runway.",
        "You can spot the A321 airliner in the satellite image.",
        "The remote sensing image shows an A321 ready for takeoff."
    ],
    'A330': [
        "An A330 is seen in the aerial image during its descent.",
        "The photograph shows an A330 jet cruising in the sky.",
        "A large A330 appears in the satellite imagery."
    ],
    'A350': [
        "An A350 can be seen in the aerial view, parked at the gate.",
        "The satellite image includes an A350 aircraft mid-flight.",
        "A distinct A350 is captured in the remote sensing picture."
    ],
    'ARJ21': [
        "An ARJ21 jetliner is visible in the aerial image.",
        "The image displays an ARJ21 on the tarmac.",
        "A clear ARJ21 is seen in the remote sensing shot."
    ],
    'Baseball': [
        "A baseball game is visible in the aerial image.",
        "The satellite image shows a baseball field with players.",
        "You can see a baseball match in the remote sensing photo."
    ],
    'Basketball': [
        "An outdoor basketball court is captured in the aerial view.",
        "The image reveals a basketball game in progress.",
        "A basketball court is seen in the satellite photograph."
    ],
    'Boeing737': [
        "A Boeing 737 is captured in the aerial image.",
        "The satellite photo shows a Boeing 737 on the runway.",
        "An aerial view featuring a Boeing 737 in flight."
    ],
    'Boeing747': [
        "A Boeing 747 is seen in the aerial photograph.",
        "The remote sensing image captures a Boeing 747 taking off.",
        "A large Boeing 747 is visible in the satellite image."
    ],
    'Boeing777': [
        "An aerial image showing a Boeing 777 on the taxiway.",
        "The photograph includes a Boeing 777 jetliner.",
        "A Boeing 777 is visible in the remote sensing shot."
    ],
    'Boeing787': [
        "A Boeing 787 can be seen in the aerial image.",
        "The satellite image features a Boeing 787 in flight.",
        "An aerial view of a Boeing 787 near the terminal."
    ],
    'Bridge': [
        "An aerial image showing a long bridge over water.",
        "The photograph includes a suspension bridge.",
        "A bridge is visible in the remote sensing shot."
    ],
    'Bus': [
        "A bus is seen in the aerial image driving down the street.",
        "The satellite image captures a bus at a bus stop.",
        "A bus is visible in the remote sensing photograph."
    ],
    'C919': [
        "An aerial shot features the C919 aircraft on the runway.",
        "You can spot the C919 jetliner in the satellite image.",
        "The remote sensing image shows a C919 ready for takeoff."
    ],
    'Cargo': [
        "A cargo ship is seen in the aerial image at the port.",
        "The photograph shows a cargo vessel sailing.",
        "A cargo ship is visible in the remote sensing image."
    ],
    'Dry': [
        "A dry cargo ship is seen in the aerial image.",
        "The satellite image shows a dry bulk carrier at sea.",
        "A dry cargo vessel is visible in the remote sensing shot."
    ],
    'Dump': [
        "A dump truck is captured in the aerial image.",
        "The photograph includes a dump truck at a construction site.",
        "A dump truck is visible in the remote sensing shot."
    ],
    'Engineering': [
        "An engineering vehicle is seen in the aerial image.",
        "The satellite photo shows an engineering truck.",
        "An engineering vehicle is visible in the remote sensing image."
    ],
    'Excavator': [
        "An excavator is captured in the aerial image working on a site.",
        "The photograph shows an excavator at a construction area.",
        "An excavator is visible in the remote sensing shot."
    ],
    'Fishing': [
        "A fishing boat is seen in the aerial image on the lake.",
        "The satellite image shows a fishing vessel in the ocean.",
        "A fishing boat is visible in the remote sensing photograph."
    ],
    'Football': [
        "A football field is seen in the aerial image.",
        "The photograph captures a football game in progress.",
        "A football stadium is visible in the satellite image."
    ],
    'Intersection': [
        "A busy intersection is seen in the aerial image.",
        "The satellite image shows cars at an intersection.",
        "An intersection is visible in the remote sensing photograph."
    ],
    'Liquid': [
        "A liquid cargo ship is seen in the aerial image at sea.",
        "The photograph shows a liquid tanker near the shore.",
        "A liquid cargo vessel is visible in the remote sensing shot."
    ],
    'Motorboat': [
        "A motorboat is seen in the aerial image speeding across the water.",
        "The satellite image shows a motorboat near the coast.",
        "A motorboat is visible in the remote sensing photograph."
    ],
    'Passenger': [
        "A passenger ship is seen in the aerial image docking.",
        "The photograph captures a passenger vessel at the pier.",
        "A passenger ship is visible in the remote sensing shot."
    ],
    'Roundabout': [
        "A roundabout is seen in the aerial image with cars circulating.",
        "The satellite image shows a busy roundabout.",
        "A roundabout is visible in the remote sensing photograph."
    ],
    'Small': [
        "A small boat is seen in the aerial image near the harbor.",
        "The photograph shows a small vessel on the lake.",
        "A small boat is visible in the remote sensing image."
    ],
    'Tennis': [
        "A tennis court is seen in the aerial image with players.",
        "The photograph shows a tennis match in progress.",
        "A tennis court is visible in the remote sensing shot."
    ],
    'Tractor': [
        "A tractor is captured in the aerial image on a farm.",
        "The satellite image shows a tractor plowing a field.",
        "A tractor is visible in the remote sensing photograph."
    ],
    'Trailer': [
        "A trailer truck is seen in the aerial image on the highway.",
        "The photograph shows a trailer at a loading dock.",
        "A trailer truck is visible in the remote sensing shot."
    ],
    'Truck': [
        "A truck is seen in the aerial image on the road.",
        "The satellite image shows a truck near a warehouse.",
        "A truck is visible in the remote sensing photograph."
    ],
    'Tugboat': [
        "A tugboat is seen in the aerial image guiding a ship.",
        "The photograph shows a tugboat in the harbor.",
        "A tugboat is visible in the remote sensing shot."
    ],
    'Van': [
        "A van is seen in the aerial image parked on the street.",
        "The satellite image shows a van near a building.",
        "A van is visible in the remote sensing photograph."
    ],
    'Warship': [
        "A warship is seen in the aerial image patrolling the sea.",
        "The photograph shows a warship in naval exercises.",
        "A warship is visible in the remote sensing shot."
    ],
    'other-airplane': [
        "An unidentified airplane is seen in the aerial image.",
        "The satellite image shows an unknown type of airplane.",
        "An unidentified aircraft is visible in the remote sensing photo."
    ],
    'other-ship': [
        "An unknown ship is seen in the aerial image at sea.",
        "The photograph shows an unidentified vessel near the coast.",
        "An unknown ship is visible in the remote sensing image."
    ],
    'other-vehicle': [
        "An unidentified vehicle is seen in the aerial image.",
        "The satellite image shows an unknown type of vehicle.",
        "An unidentified vehicle is visible in the remote sensing shot."
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








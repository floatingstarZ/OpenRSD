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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
data_root = '/data/space2/huangziyue/TGRS_HRRSD/train_val'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data4_HRRSD_New_DINOv2_VitL_gt_MLP/epoch_100.pth')
###############################################################

ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()

################################################################
from M_Tools.LabelSpace.Label3_Generate_Text_Prompts import Data4_HRRSD
phrases = Data4_HRRSD
phrases = {k.lower(): v for k, v in phrases.items()}
classes = sorted(list(phrases.keys()))
################################################################

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








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

data_root = '/data/space2/huangziyue/FMoW/train'
feat_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'

"""
from tqdm import tqdm
import os
from ctlib.os import *
data_root = '/data/space2/huangziyue/xView_800_600'
feat_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'
CLASSES = []
for feat_file in tqdm(list(os.listdir(feat_dir))):
    data = pklload(feat_dir + '/' + feat_file, msg=False)
    for name in data['cls_names']:
        if name not in CLASSES:
            CLASSES.append(name)
print(sorted(CLASSES))
"""

CLASSES = ['airport', 'airport_hangar', 'airport_terminal',
           'amusement_park', 'archaeological_site', 'border_checkpoint',
           'burial_site', 'car_dealership', 'construction_site', 'dam',
           'educational_institution', 'electric_substation',
           'factory_or_powerplant', 'fountain', 'gas_station',
           'golf_course', 'ground_transportation_station', 'helipad',
           'interchange', 'lake_or_pond', 'lighthouse', 'military_facility',
           'nuclear_powerplant', 'oil_or_gas_facility', 'park',
           'parking_lot_or_garage', 'port', 'race_track', 'railway_bridge',
           'recreational_facility', 'road_bridge', 'runway', 'shipyard',
           'shopping_mall', 'smokestack', 'solar_farm', 'space_facility',
           'stadium']



CLASSES = [c.replace('/', '_') for c in CLASSES]


id2name = {i:c for i, c in enumerate(CLASSES)}
name2id = {c:i for i, c in enumerate(CLASSES)}
# ----------------- 读取pkl数据，按照类别划分 -------------------
merge_infos = dict()
gt_feats_dict = dict()

for log_file in tqdm(sorted(list(os.listdir(feat_dir)))):
    log_data = pklload(feat_dir + '/' + log_file, msg=False)
    cls_names = log_data['cls_names']
    gt_feats = log_data['patch_feats']

    for cat_name, f_gt in zip(cls_names, gt_feats):
        cat_name = cat_name.replace('/', '_')
        if cat_name not in CLASSES: # 去掉SAM
            continue
        if cat_name not in gt_feats_dict.keys():
            gt_feats_dict[cat_name] = []
        gt_feats_dict[cat_name].append(f_gt)

# # ----------------- 类别数据采样，每一类最多2000个样本 -------------------

max_sample = 1000
sampled_gt_feats = dict()
sampled_ids = dict()
for name in gt_feats_dict.keys():
    gt_feats = gt_feats_dict[name]

    sampled_gt_feats[name] = []
    sampled_ids[name] = []
    n_total = len(gt_feats)
    n_choose = min(max_sample, n_total)
    print(f'Choose: {n_choose} / {n_total},     {name}')
    choose_ids = np.random.choice(n_total, n_choose, replace=False).tolist()

    for i in range(n_total):
        if i in choose_ids:
            sampled_gt_feats[name].append(gt_feats[i])
            sampled_ids[name].append(i)

merge_infos['sampled_gt_feats'] = sampled_gt_feats
merge_infos['sampled_ids'] = sampled_ids

# ----------------- 读取pkl数据，按照类别划分 -------------------

out_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = out_root + '/train_split.txt'
val_split_pth = out_root + '/val_split.txt'
train_gt_dir = out_root + '/train_gt'
val_gt_dir = out_root + '/val_gt'
cat_file_pth = out_root + '/categories.txt'

mkdir(out_root)
mkdir(train_gt_dir)
mkdir(val_gt_dir)

train_splits = []
val_splits = []
train_ratio = 0.5
with open(cat_file_pth, 'wt+') as f:
    for c in sampled_gt_feats.keys():
        f.write(c + '\n')

for category in sampled_gt_feats.keys():
    sampled_gts = sampled_gt_feats[category]

    train_gt_cat_dir = train_gt_dir + '/' + category
    val_gt_cat_dir = val_gt_dir + '/' + category

    mkdir(train_gt_cat_dir)
    mkdir(val_gt_cat_dir)

    n_total = len(sampled_gts)
    n_train = int(n_total * train_ratio)
    train_ids = np.random.choice(n_total, n_train, replace=False).tolist()

    for i in range(len(sampled_gts)):
        img_stem = f'{category}_{i}'

        if i in train_ids:
            train_splits.append(category + '/' + img_stem)
            pklsave(sampled_gts[i], train_gt_cat_dir + f'/{img_stem}.pkl')
        else:
            val_splits.append(category + '/' + img_stem)
            pklsave(sampled_gts[i], val_gt_cat_dir + f'/{img_stem}.pkl')

with open(train_split_pth, 'wt+') as f:
    for s in train_splits:
        f.write(s + '\n')
print(f'Save: {train_split_pth}')

with open(val_split_pth, 'wt+') as f:
    for s in val_splits:
        f.write(s + '\n')
print(f'Save: {val_split_pth}')

all_gt_feats = []
all_labels = []
for cat_name in sampled_gt_feats.keys():
    all_gt_feats.extend(sampled_gt_feats[cat_name])

    all_labels.append(np.ones(len(sampled_gt_feats[cat_name])) * name2id[cat_name])

all_gt_feats = np.stack(all_gt_feats)

all_labels = np.concatenate(all_labels).astype(np.int32)
fig_gt_pth = out_root + '/gt_feats_tsne.png'

tsne_feat = all_gt_feats
print('Len(tsne_feat):', len(tsne_feat))
tsne_label = all_labels
tsne2d = TSNE(n_components=2, init='pca', random_state=0)
X_tsne_2d = tsne2d.fit_transform(tsne_feat)

start_time = time.time()
plot_embedding_2d(X_tsne_2d, tsne_label, fig_gt_pth)
tsne_time = time.time() - start_time
print('tsne_time', tsne_time)
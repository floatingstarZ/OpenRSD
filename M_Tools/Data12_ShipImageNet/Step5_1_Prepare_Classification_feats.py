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

data_root =  '/data/space2/huangziyue/ShipRSImageNet_DOTA/train'
feat_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'


"""
from tqdm import tqdm
import os
from ctlib.os import *
data_root =  '/data/space2/huangziyue/HRSC2016_DOTA/train'
feat_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'
CLASSES = []
for feat_file in tqdm(list(os.listdir(feat_dir))):
    data = pklload(feat_dir + '/' + feat_file, msg=False)
    for name in data['cls_names']:
        if name not in CLASSES:
            CLASSES.append(name)
print(sorted(CLASSES))
"""

CLASSES = ['aoe', # 快速战斗支援舰
           'arleigh_burke_dd', # 阿利·伯克级导弹驱逐舰
           'asagiri_dd',
           'atago_dd', 'austin_ll', 'barge',
           'cargo', 'commander', 'container_ship', 'dock', 'enterprise',
           'epf', 'ferry', 'fishing_vessel', 'hatsuyuki_dd', 'hovercraft',
           'hyuga_dd', 'lha_ll', 'lsd_41_ll', 'masyuu_as', 'medical_ship',
           'midway', 'motorboat', 'nimitz', 'oil_tanker', 'osumi_ll', 'other_aircraft_carrier',
           'other_auxiliary_ship', 'other_destroyer', 'other_frigate', 'other_landing',
           'other_merchant', 'other_ship', 'other_warship', 'patrol', 'perry_ff',
           'roro', 'sailboat', 'sanantonio_as', 'submarine', 'test_ship',
           'ticonderoga', 'training_ship', 'tugboat', 'wasp_ll', 'yacht',
           'yudao_ll', 'yudeng_ll', 'yuting_ll', 'yuzhao_ll']

ship_classes_interpreter = {
    'aoe': 'Fast Combat Support Ship (AOE) # 快速战斗支援舰',
    'arleigh_burke_dd': 'Arleigh Burke-class Destroyer # 阿利·伯克级驱逐舰',
    'asagiri_dd': 'Asagiri-class Destroyer # 朝雾级驱逐舰',
    'atago_dd': 'Atago-class Destroyer # 爱宕级驱逐舰',
    'austin_ll': 'Austin-class Amphibious Transport Dock # 奥斯汀级两栖运输舰',
    'barge': 'Barge # 驳船',
    'cargo': 'Cargo Ship # 货船',
    'commander': 'Commander-class Patrol Ship # 指挥舰或巡逻舰',
    'container_ship': 'Container Ship # 集装箱船',
    'dock': 'Dock # 码头',
    'enterprise': 'USS Enterprise (Aircraft Carrier) # 企业号航空母舰',
    'epf': 'Expeditionary Fast Transport (EPF) # 远征快速运输船',
    'ferry': 'Ferry # 渡轮',
    'fishing_vessel': 'Fishing Vessel # 渔船',
    'hatsuyuki_dd': 'Hatsuyuki-class Destroyer # 初雪级驱逐舰',
    'hovercraft': 'Hovercraft # 气垫船',
    'hyuga_dd': 'Hyuga-class Helicopter Destroyer # 日向级直升机驱逐舰',
    'lha_ll': 'Landing Helicopter Assault Ship (LHA) # 两栖攻击舰',
    'lsd_41_ll': 'LSD-41 Whidbey Island-class Dock Landing Ship # 惠德贝岛级船坞登陆舰',
    'masyuu_as': 'Masyuu-class Support Ship # 真秀级支援舰',
    'medical_ship': 'Hospital Ship # 医疗船',
    'midway': 'USS Midway (Aircraft Carrier) # 中途岛号航空母舰',
    'motorboat': 'Motorboat # 摩托艇',
    'nimitz': 'Nimitz-class Aircraft Carrier # 尼米兹级航空母舰',
    'oil_tanker': 'Oil Tanker # 油轮',
    'osumi_ll': 'Osumi-class Landing Ship # 大隅级登陆舰',
    'other_aircraft_carrier': 'Other Aircraft Carrier # 其他航空母舰',
    'other_auxiliary_ship': 'Other Auxiliary Ship # 其他辅助舰',
    'other_destroyer': 'Other Destroyer # 其他驱逐舰',
    'other_frigate': 'Other Frigate # 其他护卫舰',
    'other_landing': 'Other Landing Ship # 其他登陆舰',
    'other_merchant': 'Other Merchant Ship # 其他商船',
    'other_ship': 'Other Ship # 其他船只',
    'other_warship': 'Other Warship # 其他军舰',
    'patrol': 'Patrol Ship # 巡逻船',
    'perry_ff': 'Oliver Hazard Perry-class Frigate # 奥利弗·哈扎德·佩里级护卫舰',
    'roro': 'Roll-on/Roll-off Ship (RoRo) # 滚装船',
    'sailboat': 'Sailboat # 帆船',
    'sanantonio_as': 'San Antonio-class Amphibious Transport Dock # 圣安东尼奥级两栖运输舰',
    'submarine': 'Submarine # 潜艇',
    'test_ship': 'Test Ship # 试验舰',
    'ticonderoga': 'Ticonderoga-class Cruiser # 提康德罗加级巡洋舰',
    'training_ship': 'Training Ship # 训练舰',
    'tugboat': 'Tugboat # 拖船',
    'wasp_ll': 'Wasp-class Amphibious Assault Ship # 黄蜂级两栖攻击舰',
    'yacht': 'Yacht # 游艇',
    'yudao_ll': 'Yudao-class Landing Ship # 玉岛级登陆舰',
    'yudeng_ll': 'Yudeng-class Landing Ship # 玉登级登陆舰',
    'yuting_ll': 'Yuting-class Landing Ship # 玉亭级登陆舰',
    'yuzhao_ll': 'Yuzhao-class Amphibious Transport Dock # 玉昭级两栖运输舰'
}





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

max_sample = 2000
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
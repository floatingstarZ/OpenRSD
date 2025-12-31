import cv2
import math


import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('part_id', help='devices id, 0~9')
    args = parser.parse_args()
    return args
args = parse_args()
PartID = int(args.part_id)
print('Processing Part ID:', PartID)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f'{PartID}'

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from ctlib.rbox import *
from ctlib.os import *
from ctlib.dota import load_dota
from mmrotate.structures.bbox import rbbox_overlaps
from pathlib import Path
from ctlib.rbox import obb2poly, poly2obb

src_root = '/data/space2/huangziyue/SkyScript_9_9_top50pct_filtered_by_CLIP_laion_RS'
img_dir = src_root + '/images'
ann_dir = src_root + '/annotations'

seed = 37
N = 400000
np.random.seed(seed)

out_dir = f'/data/space2/huangziyue/SkyScript_9_13_Mosaic_S{seed}_N{N}'
out_group_json = f'{out_dir}/groups.json'
out_img_dir = f'{out_dir}/images'
out_ann_dir = f'{out_dir}/annotations'
mkdir(out_dir)
mkdir(out_img_dir)
mkdir(out_ann_dir)

img_names = [Path(f).stem for f in os.listdir(img_dir)]
img_ids = np.arange(len(img_names))
img_ids = np.random.permutation(img_ids)
count = 0
Num_img = len(img_ids)

# ----- 图片分组
if not os.path.exists(out_group_json):
    img_groups = []
    start = 0
    # ----- 区块大小从 266 到 133
    candidate_M = np.array([2, 3, 4, 5, 6])
    while True:
        if count >= N:
            print(f'$$$$ Reach Maximum mosaic image: {count}')
            break
        if start >= Num_img:
            print(f'$$$$ Sampling Iteration Done, Exist mosaic image: {count}')
            start = 0
            img_ids = np.random.permutation(img_ids)
            continue
        M = int(np.random.choice(candidate_M))
        end = start + M**2
        # ----- 去掉尾部
        if end >= Num_img:
            print(f'$$$$ Sampling Iteration Done, Exist mosaic image: {count}')
            start = 0
            img_ids = np.random.permutation(img_ids)
            continue

        group_names = []
        for i in img_ids[start: end]:
            group_names.append(img_names[i])
        img_groups.append(group_names)
        count += 1
        start += M**2

    jsonsave(img_groups, out_group_json)
#########################################
img_groups = jsonload(out_group_json)
print(img_groups[0])
H, W = 512, 512
a = int(512 * 1 / 16)
b = int(512 * 15 / 16)
out_H, out_W = 800, 800

n_img = len(img_groups)
group_ids = list(range(n_img))
num_per_part = n_img // 8
img_groups = img_groups[num_per_part * PartID: num_per_part * (PartID+1)]
group_ids = group_ids[num_per_part * PartID: num_per_part * (PartID+1)]

for group_id, img_names in tqdm(list(zip(group_ids, img_groups))):

    n_img = len(img_names)
    M = int(math.sqrt(n_img))
    assert M**2 == n_img

    out_name = f'G{group_id}_M{M}'
    out_ann_pth = f'{out_ann_dir}/{out_name}.pkl'
    out_img_pth = f'{out_img_dir}/{out_name}.png'
    if os.path.exists(out_ann_pth) and os.path.exists(out_img_pth):
        continue

    images = []
    polys = []
    texts = []
    text_embeds = []
    visual_embeds = []
    for name in img_names:
        img_pth = f'{img_dir}/{name}.png'
        ann_pth = f'{ann_dir}/{name}.pkl'

        img = cv2.imread(img_pth)
        img = cv2.resize(img, (512, 512))
        ann_info = pklload(ann_pth, msg=False)
        texts.append(ann_info['texts'])
        text_embeds.append(ann_info['texts_embeds'])
        visual_embeds.append(ann_info['visual_embeds'])
        images.append(img)
        # 512 / 8 = 64, 512 / 8 * 7 = 448
        polys.append(np.array([[a, a, b, a, b, b, a, b]]))
    text_embeds = np.concatenate(text_embeds)
    visual_embeds = np.concatenate(visual_embeds)
    polys = np.concatenate(polys)

    spliced_img = np.zeros((H * M, W * M, 3), np.uint8)
    for i in range(M):
        for j in range(M):
            spliced_img[i * H: (i + 1) * H, j * W: (j + 1) * W] = images[i * M + j]
            polys[i * M + j, 0::2] = polys[i * M + j, 0::2] + j * W
            polys[i * M + j, 1::2] = polys[i * M + j, 1::2] + i * W

    s_H, s_W = spliced_img.shape[:2]
    polys[:, 0::2] = polys[:, 0::2] * out_H / s_H
    polys[:, 1::2] = polys[:, 1::2] * out_W / s_W

    spliced_img = cv2.resize(spliced_img, (out_H, out_W))


    # ----- 保存结果
    cv2.imwrite(out_img_pth, spliced_img)
    pklsave(dict(
        texts=texts,
        text_embeds=text_embeds,
        visual_embeds=visual_embeds,
        polys=polys
    ), out_ann_pth, msg=False)




































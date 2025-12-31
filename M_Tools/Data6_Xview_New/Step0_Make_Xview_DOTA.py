from ctlib.os import *
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
# https://github.com/DIUx-xView/xView1_baseline

"""
手动删掉了1395的txt，因为没有图片
"""

data_root = '/data/space2/huangziyue/xView'
image_dir = f'{data_root}/train_images'
ann_file_pth = f'{data_root}/xView_train.geojson'
category_id_to_name = {}
with open(f"{data_root}/xview_class_labels_new.txt", encoding="utf8") as f:
    lines = f.readlines()
for line in lines:
    category_id = line.split(":")[0]
    category_name = line.split(":")[1].replace("\n", "")
    category_name = category_name.replace(' ', '_')
    category_id_to_name[int(category_id)] = category_name

ann_file = jsonload(ann_file_pth)

ann_infos = dict()
for instance_ann in tqdm(ann_file['features']):
    cat_id = instance_ann['properties']['type_id']
    box_str = instance_ann['properties']['bounds_imcoords']
    box = [int(s) for s in box_str.split(',')]
    img_file = instance_ann['properties']['image_id']
    if img_file not in ann_infos.keys():
       ann_infos[img_file] = dict(cls_names=[], boxes=[])
    if cat_id not in category_id_to_name.keys():
        print(f'{cat_id} is not valid in cat_dict')
        continue
    ann_infos[img_file]['cls_names'].append(category_id_to_name[cat_id])
    ann_infos[img_file]['boxes'].append(box)

    a = 0
out_ann_dir = f'{data_root}/labelTxt'
mkdir(out_ann_dir)
for img_file, ann_info in tqdm(list(ann_infos.items())):
    if len(ann_info['boxes']) == 0:
        continue
    try:
        img = cv2.imread(f'{image_dir}/{img_file}')
    except:
        # 1395 missing
        print(f'Invalid Image: {img_file}')
        continue
    img_name = Path(img_file).stem
    out_ann_file = f'{out_ann_dir}/{img_name}.txt'
    with open(out_ann_file, 'wt+') as f:
        for box, class_name in zip(ann_info['boxes'], ann_info['cls_names']):
            x1, y1, x2, y2 = box
            f.write(f'{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2} {class_name} 0 \n')




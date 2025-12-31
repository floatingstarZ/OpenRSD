import os
from ctlib.os import *
import cv2
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from PIL import Image
import mmcv

# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
import torch
from ctlib.rbox import *
from tqdm import tqdm
from pathlib import Path

data_root = '/data/space2/huangziyue/ShipRSImageNet_V1/VOC_Format'
img_dir = data_root + '/JPEGImages'
ann_dir = data_root + '/Annotations'
split_file = data_root + '/ImageSets/train.txt'

out_root = '/data/space2/huangziyue/ShipRSImageNet_DOTA'
mkdir(out_root)
out_root = out_root + '/train'
out_img_dir = f'{out_root}/images'
out_ann_dir = f'{out_root}/labelTxt'
mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)

with open(split_file, 'r') as f:
    img_ids = list([Path(l.strip()).stem for l in f.readlines()])

for img_id in tqdm(img_ids):
    data_info = {}

    img_file_pth = f'{img_dir}/{img_id}.bmp'
    xml_path = f'{ann_dir}/{img_id}.xml'

    if not os.path.exists(img_file_pth):
        print(f'Missing {img_file_pth}')
        continue
    if not os.path.exists(xml_path):
        print(f'Missing {xml_path}')
        continue
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_bboxes = []
    gt_names = []
    gt_polys = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        name = name.lower().strip().replace(' ', '_').replace('-', '_')

        poly_obj = obj.find('polygon')

        # Add an extra score to use obb2poly_np
        poly = [
            float(poly_obj.find('x1').text),
            float(poly_obj.find('y1').text),
            float(poly_obj.find('x2').text),
            float(poly_obj.find('y2').text),
            float(poly_obj.find('x3').text),
            float(poly_obj.find('y3').text),
            float(poly_obj.find('x4').text),
            float(poly_obj.find('y4').text),
        ]
        gt_polys.append(poly)
        gt_names.append(name)

    img = cv2.imread(img_file_pth)
    out_img_pth = f'{out_img_dir}/{img_id}.png'
    out_ann_pth = f'{out_ann_dir}/{img_id}.txt'
    cv2.imwrite(out_img_pth, img)

    with open(out_ann_pth, 'wt+') as f:
        for name, poly in zip(gt_names, gt_polys):
            for coord in poly:
                f.write('%.1f ' % (coord))
            f.write(f'{name} 0 \n')


a = 0
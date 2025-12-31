import os
from ctlib.os import *
import cv2
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from PIL import Image
import mmcv

HRSC_CLASS = ('ship', )
valid_classes_in_train = ['Arleigh_Burke', 'Austen', 'Car_carrier', 'CntShip',
                           'Container', 'Cruise', 'Enterprise', 'Hovercraft',
                           'Kuznetsov', 'Medical', 'Midway_class', 'Nimitz',
                           'OXo', 'Perry', 'Sanantonio', 'Tarawa', 'Ticonderoga',
                           'WhidbeyIsland', 'aircraft_carrier', 'lute', 'merchant_ship',
                           'ship', 'submarine', 'warcraft', 'yacht']

HRSC_CLASSES = ('ship', 'aircraft_carrier', 'warcraft', 'merchant_ship',
                'Nimitz', 'Enterprise', 'Arleigh_Burke', 'WhidbeyIsland',
                'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty_Hawk',
                'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue_Ridge',
                'Container', 'OXo', 'Car_carrier',
                'Hovercraft', 'yacht', 'CntShip', 'Cruise',
                'submarine', 'lute', 'Medical', 'Car_carrier',
                'Ford-_class', 'Midway_class', 'Invincible_class')
HRSC_CLASSES_ID = ('01', '02', '03', '04', '05', '06', '07', '08', '09',
                   '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', '20', '22', '24', '25', '26', '27', '28', '29',
                   '30', '31', '32', '33')
# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
import torch
from ctlib.rbox import *
from tqdm import tqdm

data_root = '/data/space2/huangziyue/HRSC2016'
img_dir = data_root + '/FullDataSet/AllImages'
ann_dir = data_root + '/FullDataSet/Annotations'
split_file = data_root + '/ImageSets/test.txt'

out_root = '/data/space2/huangziyue/HRSC2016_DOTA'
mkdir(out_root)
out_root = out_root + '/test'
out_img_dir = f'{out_root}/images'
out_ann_dir = f'{out_root}/labelTxt'
mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)

catid2label = {
    ('1' + '0' * 6 + cls_id): i
    for i, cls_id in enumerate(HRSC_CLASSES_ID)
}

catid2name = {
    ('1' + '0' * 6 + cls_id): cls_name
    for cls_name, cls_id in zip(HRSC_CLASSES, HRSC_CLASSES_ID)
}

with open(split_file, 'r') as f:
    img_ids = list([l.strip() for l in f.readlines()])

for img_id in tqdm(img_ids):
    data_info = {}

    img_file_pth = f'{img_dir}/{img_id}.bmp'
    xml_path = f'{ann_dir}/{img_id}.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_bboxes = []
    gt_names = []

    for obj in root.findall('HRSC_Objects/HRSC_Object'):
        class_id = obj.find('Class_ID').text
        name = catid2name.get(class_id)
        if name is None:
            print(f'[ERROR] {class_id} is not exist')
            continue
        if name not in valid_classes_in_train:
            print(f'[ERROR] {name} not in training')
            continue

        # Add an extra score to use obb2poly_np
        bbox = [
            float(obj.find('mbox_cx').text),
            float(obj.find('mbox_cy').text),
            float(obj.find('mbox_w').text),
            float(obj.find('mbox_h').text),
            float(obj.find('mbox_ang').text)
        ]

        gt_bboxes.append(bbox)
        gt_names.append(name)
    gt_bboxes = np.array(gt_bboxes)
    gt_polys = obb2poly(gt_bboxes)


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
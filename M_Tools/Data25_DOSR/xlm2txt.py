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

xlm_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data25_DOSR/Annotations_8_parameters_version'
txt_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data25_DOSR/labelTxt'
mkdir(txt_dir)

for xml_file in tqdm(os.listdir(xlm_dir)):
    file_name = Path(xml_file).stem
    data_info = {}

    xml_path = f'{xlm_dir}/{xml_file}'
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_bboxes = []
    gt_names = []
    gt_polys = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        name = name.lower().strip().replace(' ', '_').replace('-', '_')

        poly_obj = obj.find('bndbox')

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
    out_ann_pth = f'{txt_dir}/{file_name}.txt'
    with open(out_ann_pth, 'wt+') as f:
        for name, poly in zip(gt_names, gt_polys):
            for coord in poly:
                f.write('%.1f ' % (coord))
            f.write(f'{name} 0 \n')


a = 0
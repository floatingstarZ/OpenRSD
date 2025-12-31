from ctlib.os import *
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
# https://github.com/DIUx-xView/xView1_baseline
data_root = '/data/space2/huangziyue/xView'
img_dir = f'{data_root}/train_images'
ann_dir = f'{data_root}/labelTxt'

# 146测试，700训练
np.random.seed(2024)
ann_files = np.random.permutation(np.array(os.listdir(ann_dir)))
test_files = sorted(ann_files[:146].tolist())
train_files = sorted(ann_files[146:].tolist())

for files, split in zip([train_files, test_files], ['train', 'test']):
    split_data_root = f'{data_root}/{split}'
    mkdir(split_data_root)
    split_img_dir = f'{split_data_root}/images'
    split_ann_dir = f'{split_data_root}/labelTxt'
    mkdir(split_ann_dir)
    mkdir(split_img_dir)
    file_split = [Path(f).stem for f in files]
    split_file_pth = f'{split_data_root}/split.txt'
    with open(split_file_pth, 'wt+') as f:
        for file in file_split:
            f.write(f'{file}\n')
    for file in tqdm(file_split):
        src_ann_pth = f'{ann_dir}/{file}.txt'
        tgt_ann_pth = f'{split_ann_dir}/{file}.txt'
        src_img_pth = f'{img_dir}/{file}.tif'
        tgt_img_pth = f'{split_img_dir}/{file}.tif'
        os.system(f'cp {src_ann_pth} {tgt_ann_pth}')
        os.system(f'cp {src_img_pth} {tgt_img_pth}')

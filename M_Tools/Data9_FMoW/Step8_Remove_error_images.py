from ctlib.os import *
import os
from pathlib import Path
from tqdm import tqdm

data_root = '/data/space2/huangziyue/FMoW/test'
img_dir = f'{data_root}/images'
ann_dir = f'{data_root}/labelTxt'
error_files = []
for img_file in tqdm(os.listdir(img_dir)):
    try:
        img = cv2.imread(f'{img_dir}/{img_file}')
        if img is None:
            img_name = Path(img_file).stem
            print(f'Error: {img_name}, Removed')
            error_files.append(img_name)
            img_pth = f'{img_dir}/{img_file}'
            ann_pth = f'{ann_dir}/{img_name}.txt'
            os.remove(img_pth)
            os.remove(ann_pth)
    except:
        img_name = Path(img_file).stem
        print(f'Error: {img_name}, Removed')
        error_files.append(img_name)
        img_pth = f'{img_dir}/{img_file}'
        ann_pth = f'{ann_dir}/{img_name}.txt'
        os.remove(img_pth)
        os.remove(ann_pth)
with open(f'{data_root}/error_removed_files.txt', 'wt+') as f:
    for img_name in error_files:
        f.write(f'{img_name}\n')

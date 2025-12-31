import os
from ctlib.os import *
from tqdm import tqdm
from pathlib import Path
data_root = '/data/space2/huangziyue/spacenet'
out_root = '/data/space2/huangziyue/Spacenet_Merge'
out_img_dir = f'{out_root}/images'
out_ann_dir = f'{out_root}/annotations'

img_dirs = [
    '/data/space2/huangziyue/spacenet/fusion_3band/train/JPEGImages', # AOI_2_Vegas
    '/data/space2/huangziyue/spacenet/1_Rio/dstdata/train/JPEGImages', # AOI_1_Rio
    '/data/space2/huangziyue/spacenet/AOI_3_Paris_Train/train/JPEGImages_png',
    '/data/space2/huangziyue/spacenet/AOI_4_Shanghai_Train/train/JPEGImages_png',
    '/data/space2/huangziyue/spacenet/AOI_5_Khartoum_Train/train/JPEGImages_png',
]

ann_dirs = [
    '/data/space2/huangziyue/spacenet/fusion_3band/train/labelTxt',
    '/data/space2/huangziyue/spacenet/1_Rio/dstdata/train/labelTxt',
    '/data/space2/huangziyue/spacenet/AOI_3_Paris_Train/train/labelTxt',
    '/data/space2/huangziyue/spacenet/AOI_4_Shanghai_Train/train/labelTxt',
    '/data/space2/huangziyue/spacenet/AOI_5_Khartoum_Train/train/labelTxt'
]
mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)

for img_dir, ann_dir in zip(img_dirs, ann_dirs):
    for ann_file in tqdm(list(os.listdir(ann_dir))):
        ann_pth = f'{ann_dir}/{ann_file}'
        img_name = Path(ann_file).stem
        img_pth = f'{img_dir}/{img_name}.png'
        if not os.path.exists(img_pth):
            print(f'Error: {img_pth} does not exists')

        tgt_ann_pth = f'{out_ann_dir}/{ann_file}'
        tgt_img_pth = f'{out_img_dir}/{img_name}.png'
        os.system(f'cp {ann_pth} {tgt_ann_pth}')
        os.system(f'cp {img_pth} {tgt_img_pth}')


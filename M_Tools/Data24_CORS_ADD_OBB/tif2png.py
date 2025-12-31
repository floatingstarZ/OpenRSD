import os
import cv2
from pathlib import Path
from ctlib.os import *
from tqdm import tqdm

img_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB/images/train2017'
out_img_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB/images/train2017_png'
mkdir(out_img_dir)
for img_file in tqdm(list(os.listdir(img_dir))):
    img_name = Path(img_file).stem
    img_pth = f'{img_dir}/{img_file}'
    out_pth = f'{out_img_dir}/{img_name}.png'
    img = cv2.imread(img_pth)
    cv2.imwrite(out_pth, img)

img_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB/images/val2017'
out_img_dir = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB/images/val2017_png'
mkdir(out_img_dir)
for img_file in tqdm(list(os.listdir(img_dir))):
    img_name = Path(img_file).stem
    img_pth = f'{img_dir}/{img_file}'
    out_pth = f'{out_img_dir}/{img_name}.png'
    img = cv2.imread(img_pth)
    cv2.imwrite(out_pth, img)

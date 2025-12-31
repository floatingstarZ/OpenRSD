import os
import cv2
from ctlib.os import *
from pathlib import Path

img_root = '/data/space2/huangziyue/SkyScript'
#######################

split = 'images7'
img_dir = f'{img_root}/{split}'
vid_dir = f'{img_root}/Step0_vis_high_resolution'
mkdir(vid_dir)

N = 0
for img_file in os.listdir(img_dir):
    if N > 1000:
        break
    try:
        img_pth = img_dir + '/' + img_file
        img = cv2.imread(img_pth)
        H, W, C = img.shape
        if H < 512 or W < 512:
            continue
        img_stem = Path(img_pth).stem
        print('img_stem', img_stem)
        cv2.imwrite(f'{vid_dir}/{img_stem}.png', img)
    except:
        print(f'Error in {img_file}')


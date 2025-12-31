import os

import cv2

PartID = 7
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f'{PartID}'

from pathlib import Path
from commonlibs.common_tools import *
import open_clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from PIL import Image
import clip
import open_clip
from torch.utils.data import DataLoader, Dataset
import io
import random
import argparse
import pandas as pd
from tqdm import tqdm

import time
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Compose
import numpy as np
from ctlib.os import *
from pathlib import Path
import torch.nn.functional as F
from ctlib.transform import to_array

Image.MAX_IMAGE_PIXELS = None

from SkyScript_open_clip.factory import create_model_and_transforms, \
    create_model_from_pretrained, create_model

def model_forward(clip_model, images, batch_size=64):
    img_features = []
    for i in tqdm(list(range(len(images) // batch_size + 1))):
        if i*batch_size == len(images):
            continue
        part_images = images[i*batch_size: (i+1)*batch_size, ...]
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_feats = clip_model.encode_image(part_images.cuda())
            image_feats /= image_feats.norm(dim=-1, keepdim=True)
            img_features.append(image_feats)
    img_features = torch.cat(img_features, dim=0)
    return img_features.detach().cpu()

img_root = '/data/space2/huangziyue/SkyScript'
#######################

split = 'images7'
img_dir = f'{img_root}/{split}'

out_embed_dir = f'{img_root}/7_8_SkyCLIP_ViT_L_embed_{split}'
mkdir(out_embed_dir)
#######################


ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()

main_caption_file = '/data/space2/huangziyue/SkyScript/SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv'
data_root = '/data/space2/huangziyue/SkyScript'

out_root = '/data/space2/huangziyue/SkyScript_9_9_top50pct_filtered_by_CLIP_laion_RS'
out_img_dir = f'{out_root}/images'
out_ann_dir = f'{out_root}/annotations'
mkdir(out_root)
mkdir(out_img_dir)
mkdir(out_ann_dir)
N = 128 * 100000000

#### 分Part
df = pd.read_csv(main_caption_file)
n_captions = len(df)
sample_ids = list(range(n_captions))
num_per_part = n_captions // 8
sample_ids = sample_ids[num_per_part * PartID: num_per_part * (PartID+1)]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

visual_model = torch.hub.load(
    repo_or_dir='/home/huangziyue/.cache/torch/hub/facebookresearch_dinov2_main',
    model='dinov2_vitl14_reg',
    source='local'
)
visual_model.cuda().eval()

for index in tqdm(sample_ids):
    if index > N:
        break

    row = df.iloc[index].to_dict()

    img_dir = os.path.dirname(row['filepath'])
    filepath = row['filepath']
    img_pth = f'{data_root}/{filepath}'
    img_stem = Path(img_pth).stem
    out_name = f'{img_dir}_{img_stem}'
    out_ann_pth = f'{out_ann_dir}/{out_name}.pkl'
    out_img_pth = f'{out_img_dir}/{out_name}.png'
    if os.path.exists(out_ann_pth):
        print(f'Pass: {img_pth}')
        continue

    #################### 提取图像特征 ################
    try:
        img_name = Path(img_pth).stem
        img = Image.open(str(img_pth)).convert("RGB")
        img = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = visual_model(img)
        image_features = to_array(image_features)
    except:
        print(f'Error_Read: {img_pth}')
        continue
    ################# 转移图像 ####################
    img = cv2.imread(img_pth)
    if img is None:
        print(f'Error_Read: {img_pth}')
        continue
    cv2.imwrite(out_img_pth, img)

    ################# 提取文本特征 ####################
    caption = row["title"]
    multi_caption = row['title_multi_objects']
    text = tokenizer([caption,])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.cuda()).detach().cpu()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeds = to_array(text_features)


    out_info = dict(
        texts=[caption, ],
        multi_texts=[multi_caption,],
        texts_embeds=text_embeds,
        visual_embeds=image_features
    )
    pklsave(out_info, out_ann_pth, msg=False)






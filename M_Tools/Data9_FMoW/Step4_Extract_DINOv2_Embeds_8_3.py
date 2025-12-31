# Copyright 2023 Bytedance Ltd. and/or its affiliates

PartID = 8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f'{PartID}'

import os.path

import torch

from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmdet.structures.bbox import bbox2roi
from mmrotate.registry import MODELS
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from commonlibs.common_tools import *
import open_clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def to_array(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    elif type(x) == np.array:
        return x
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x

def imdenormalize(img, mean, std):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = img * std
    img = img + mean
    return img

def auto_denorm(img):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img = img.transpose(1, 2, 0)
    img = imdenormalize(img.astype(np.float32),
                        np.array(mean),
                        np.array(std))
    img = img.clip(min=0, max=255)
    img = img.astype(np.uint8)
    out_img = np.ascontiguousarray(img)
    return out_img

def auto_denorm_multi(imgs):
    out_imgs = []
    for img in imgs:
        out_imgs.append(auto_denorm(img))
    return np.array(out_imgs)

import torch.nn as nn
from mmdet.models.detectors import TwoStageDetector
from copy import deepcopy




def model_forward(model, images, batch_size=64):
    img_features = []
    for i in list(range(len(images) // batch_size + 1)):
        if i*batch_size == len(images):
            continue
        with torch.no_grad():
            part_images = images[i*batch_size: (i+1)*batch_size, ...]
            part_feats = model(part_images)
            out_patch_feats = deepcopy(part_feats.detach().cpu())
            del part_feats, part_images
            img_features.append(out_patch_feats)
    img_features = torch.cat(img_features, dim=0)
    return img_features


from pyiqa.utils.color_util import to_y_channel

def entropy(x, data_range=255., eps=1e-8, color_space='yiq'):
    r"""Compute entropy of a gray scale image.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
    Returns:
        Entropy of the image.
    """

    if (x.shape[1] == 3):
        # Convert RGB image to gray scale and use Y-channel
        x = to_y_channel(x, data_range, color_space)

    # Compute histogram
    hist = nn.functional.one_hot(x.long(), num_classes=int(data_range + 1)).sum(dim=[1, 2, 3])
    hist = torch.tensor(hist).to(x.device)
    hist = hist / hist.sum(dim=1, keepdim=True)

    # Compute entropy
    score = -torch.sum(hist * torch.log2(hist + eps), dim=1)

    return score

def entropy_forward(images, batch_size):
    entropys = []
    for i in list(range(len(images) // batch_size + 1)):
        if i*batch_size == len(images):
            continue
        with torch.no_grad():
            part_images = images[i*batch_size: (i+1)*batch_size, ...]
            e = entropy(part_images)
            entropys.append(e)
    entropys = torch.cat(entropys)
    return entropys

class ToolFeatExtractor(nn.Module):
    def __init__(self,
                 out_dir,
                 bbox_roi_extractor=dict(
                     out_size=(224, 224),  # (224, 224),
                     spatial_scale=1.0,
                     sampling_ratio=2,
                     clockwise=True
                 ),
                 entropy_bbox_roi_extractor=dict(
                     out_size=(64, 64),  # (224, 224),
                     spatial_scale=1.0,
                     sampling_ratio=2,
                     clockwise=True
                 ),
                 roi_scale_factor=1.25):
        super().__init__()
        self.roi_align = RoIAlignRotated(**bbox_roi_extractor)
        self.roi_align_entropy = RoIAlignRotated(**entropy_bbox_roi_extractor)
        # -对roi进行适当放大，获得一些Context信息
        self.roi_scale_factor = roi_scale_factor
        # -保留roi的宽高比，使得Crop的图像中，物体的宽高比信息得以保留
        # -保留ROI的宽高比通过以下方式实现
        # 1. 取ROI的w,h的最大值，并将短边设置为最大值，获得正方形ROI
        # 2. 由于部分物体宽高比过于悬殊，这种正方形ROI很容易引入过多的Context信息，
        #    扰乱特征的学习（可以使用padding解决该问题，但是也比较麻烦。）
        self.roi_keep_as_ratio = True
        self.pooling = torch.nn.Sequential(
            torch.nn.AvgPool2d((7, 7)),
            torch.nn.Flatten(1)
        )
        ############
        self.out_dir = out_dir
        mkdir(self.out_dir)

        self.backbone = torch.hub.load(
            repo_or_dir='/home/huangziyue/.cache/torch/hub/facebookresearch_dinov2_main',
            model='dinov2_vitl14_reg',
            source='local'
        )
        self.backbone.eval()
        self.backbone._is_init = True

        self.entropy_thr = 5.8
        self.min_area = 8 * 8

    ######################################
    def init_weights(self) -> None:
        self._is_init = True
    ######################################

    def crop(self, img, rois):
        """

        :param img: Tensor, (B,C,H,W)
        :param rois: Tensor, (N, 6)
        :return: patches: Tensor, (N, C, H_roi, W_roi)
        """
        # -------- scale --------
        h_scale_factor, w_scale_factor = self.roi_scale_factor, self.roi_scale_factor
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        # -------- keep ratio --------
        if self.roi_keep_as_ratio:
            max_wh = torch.max(new_rois[:, 3:5], dim=1)[0]
            new_rois[:, 3] = max_wh
            new_rois[:, 4] = max_wh
        patches = self.roi_align(img, new_rois)
        return patches

    def crop_ent(self, img, rois):
        """

        :param img: Tensor, (B,C,H,W)
        :param rois: Tensor, (N, 6)
        :return: patches: Tensor, (N, C, H_roi, W_roi)
        """
        # -------- scale --------
        h_scale_factor, w_scale_factor = self.roi_scale_factor, self.roi_scale_factor
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        # -------- keep ratio --------
        if self.roi_keep_as_ratio:
            max_wh = torch.max(new_rois[:, 3:5], dim=1)[0]
            new_rois[:, 3] = max_wh
            new_rois[:, 4] = max_wh
        patches = self.roi_align_entropy(img, new_rois)
        return patches

    def loss(self,
             batch_inputs,
             filename,
             gt_bboxes,
             gt_names,
             filter_using_entropy_and_area=False):
        global_idx = torch.arange(len(gt_names)).to(batch_inputs.device)
        gt_names = np.array(gt_names)
        out_data_pth = self.out_dir + '/' + filename + '.pkl'
        if os.path.exists(out_data_pth):
            return

        rois = bbox2roi(gt_bboxes)
        if filter_using_entropy_and_area:
            ######## --- 面积筛选
            areas = rois[:, 2] * rois[:, 3]
            # print('area', torch.sum(areas < self.min_area))
            valid_ids = areas >= self.min_area
            rois = rois[valid_ids]
            global_idx = global_idx[valid_ids]
            if torch.sum(valid_ids) == 0:
                return

            ######## --- 图像熵筛选
            device = rois.device
            entropy_img_patches = self.crop_ent(batch_inputs, rois)
            entropy_img_patches = auto_denorm_multi(to_array(entropy_img_patches))
            entropy_img_patches = torch.tensor(entropy_img_patches).to(device)
            img_quality = entropy_forward(entropy_img_patches, batch_size=64)
            img_quality = img_quality.to(device)
            valid_ids = img_quality >= self.entropy_thr
            # print('img_quality', torch.sum(img_quality < self.entropy_thr))
            img_quality = img_quality[valid_ids]
            rois = rois[valid_ids]
            global_idx = global_idx[valid_ids]
            if torch.sum(valid_ids) == 0:
                return
        img_patches = self.crop(batch_inputs, rois)
        patch_feats = model_forward(self.backbone, img_patches, batch_size=128)

        gt_names = gt_names[global_idx.detach().cpu().numpy()].tolist()
        out_data = dict(
            cls_names=gt_names,
            rboxes=to_array(rois[:, 1:]),
            patch_feats=to_array(patch_feats),
        )
        pklsave(out_data, out_data_pth)
        return



from ctlib.rbox import *

data_root = '/data/space2/huangziyue/FMoW/train'
out_dir = f'{data_root}/Step4_Extract_DINOv2_Embeds_8_3_GT'
model = ToolFeatExtractor(out_dir=out_dir, roi_scale_factor=1.25)
model.cuda().eval()
img_dir = f'{data_root}/images'
ann_dir = f'{data_root}/labelTxt'
file_list = sorted(os.listdir(ann_dir))
########


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
for ann_file in tqdm(file_list):
    img_name = Path(ann_file).stem
    img_pth = f'{img_dir}/{img_name}.png'

    img = Image.open(str(img_pth)).convert("RGB")
    img = transform(img).unsqueeze(0).cuda()

    ann_pth = f'{ann_dir}/{ann_file}'
    with open(ann_pth) as f:
        lines = f.readlines()
        lines = [l.strip().split(' ') for l in lines]
    if len(lines) == 0:
        print(f'Skipping {img_name}, no annotations')
        continue
    gt_polys = []
    gt_names = []
    for l in lines:
        poly = [float(coord) for coord in l[:8]]
        poly = np.array(poly)
        gt_polys.append(poly)
        gt_names.append(l[8])
    gt_polys = np.array(gt_polys)
    gt_rboxes = poly2obb(gt_polys)
    gt_rboxes = torch.tensor(gt_rboxes).cuda()
    model.loss(img, img_name, [gt_rboxes,], gt_names,
               filter_using_entropy_and_area=False)





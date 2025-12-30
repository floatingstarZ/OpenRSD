# Copyright 2023 Bytedance Ltd. and/or its affiliates
import os.path

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

def imdenormalize(img, mean, std, to_bgr=True):
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
from mmdet.models.backbones import SwinTransformer
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

@MODELS.register_module()
class ToolFeatExtractor(TwoStageDetector):
    def __init__(self,
                 out_dir,
                 bbox_roi_extractor=dict(
                     out_size=(224, 224),  # (224, 224),
                     spatial_scale=1.0,
                     sampling_ratio=2,
                     clockwise=True
                 ),
                 roi_scale_factor=1.25,
                 *args,
                 **kwargs):
        super(ToolFeatExtractor, self).__init__(*args, **kwargs)
        self.roi_align = RoIAlignRotated(**bbox_roi_extractor)
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
        # -------- use_meta_conv，结合mask来提取特征 -> concate with roi masks: 3 channel -> 4 channel  ---------
        device = rois.device
        N, C, W, H = patches.shape
        mask_w = (torch.ones(N) * W).to(device) / w_scale_factor
        mask_h = (torch.ones(N) * H).to(device) / h_scale_factor
        if self.roi_keep_as_ratio:
            max_wh = torch.max(rois[:, 3:5], dim=1)[0]
            mask_w = mask_w * (rois[:, 3] / max_wh)
            mask_h = mask_h * (rois[:, 4] / max_wh)
        masks = []
        boxes = []
        for w, h in zip(mask_w, mask_h):
            x1 = int(H / 2 - h / 2)
            y1 = int(W / 2 - w / 2)
            x2 = int(H / 2 + h / 2)
            y2 = int(W / 2 + w / 2)
            mask = torch.zeros([W, H]).to(device)
            mask[x1:x2, y1:y2] = 1
            masks.append(mask[None, ...])
            boxes.append([x1, y1, x2, y2])
        return patches, boxes

    def loss(self,
             batch_inputs,
             batch_data_samples):
        img_metas = [sample.metainfo for sample in batch_data_samples]
        filename = Path(img_metas[0]['img_path']).stem
        out_data_pth = self.out_dir + '/' + filename + '.pkl'
        if os.path.exists(out_data_pth):
            losses = dict()
            loss_out = sum([x.view(-1)[0] for x in self.parameters()]) * 0.
            losses['loss_out'] = loss_out
            return losses
        gt_bboxes = [sample.gt_instances.bboxes for sample in batch_data_samples]
        gt_labels = [sample.gt_instances.labels for sample in batch_data_samples]

        rois = bbox2roi(gt_bboxes)
        img_patches, patch_boxes = self.crop(batch_inputs, rois)
        patch_feats = model_forward(self.backbone, img_patches, batch_size=128)

        out_data = dict(
            img_metas=img_metas,
            rois=to_array(rois),
            patch_boxes=to_array(patch_boxes),
            gt_labels=to_array(gt_labels),
            patch_feats=to_array(patch_feats),
        )
        pklsave(out_data, out_data_pth)

        losses = dict()

        loss_out = sum([x.view(-1)[0] for x in self.parameters()]) * 0.
        losses['loss_out'] = loss_out
        return losses





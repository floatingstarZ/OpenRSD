# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.backbones.cspnext import CSPNeXt
from mmrotate.registry import MODELS
import mmdet.registry as mmd_registry
from mmdet.models.backbones.resnet import ResNet

@MODELS.register_module()
class GraftCSPNeXt_ResNet(CSPNeXt):
    """CSPNeXt backbone used in RTMDet.
    Meta-ResNet used in Meta-RCNN
    """
    def __init__(self,
                 arch,
                 widen_factor,
                 norm_cfg,
                 act_cfg,
                 #############
                 graft_backbone,
                 graft_pre_layer,   # cspnext的层数（哪层输出）
                 graft_after_layer, # resnet的层数（哪层输入）
                 graft_mapping,
                 use_graft=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(arch=arch,
                         widen_factor=widen_factor,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         *args,
                         **kwargs)
        arch_setting = self.arch_settings[arch]
        # ---- resnet
        self.graft_backbone = mmd_registry.MODELS.build(graft_backbone)
        self.graft_pre_layer = graft_pre_layer
        self.graft_after_layer = graft_after_layer
        self.use_graft = use_graft

        in_c = graft_mapping[0]
        out_c = graft_mapping[1]
        kernel_size = graft_mapping[2]
        stride = graft_mapping[3]
        padding = graft_mapping[4]
        if self.use_graft in [True]:
            self.graft_mapping = nn.Conv2d(in_c, out_c,
                                           kernel_size, stride, padding)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        # ----- 避免没有梯度反传
        x = x + 0.0 * sum([x.view(-1)[0] for x in self.graft_backbone.parameters()])
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            # print(i, x.shape)
            """
            M: 
            0 torch.Size([2, 48, 400, 400])
            1 torch.Size([2, 96, 200, 200])
            2 torch.Size([2, 192, 100, 100])
            3 torch.Size([2, 384, 50, 50])
            4 torch.Size([2, 768, 25, 25])
            """
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward_embed(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        # # ----------- Check Layers
        if self.use_graft == 'test':
            return self.graft_backbone(x)

        if not self.use_graft:
            x = self.graft_backbone.conv1(x)
            x = self.graft_backbone.norm1(x)
            x = self.graft_backbone.relu(x)
            x = self.graft_backbone.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.graft_backbone.res_layers):
                # print(x.shape)
                res_layer = getattr(self.graft_backbone, layer_name)
                x = res_layer(x)
            outs.append(x)
            return tuple(outs)

        ###################################
        # ----- CSPNext
        pre_layers = self.layers[:self.graft_pre_layer]
        for i, layer_name in enumerate(pre_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        # print('Pre Graft X', x.shape)
        x = self.graft_mapping(x)
        # ----- ResNet
        res_layers = self.graft_backbone.res_layers[self.graft_after_layer:]
        for i, layer_name in enumerate(res_layers):
            res_layer = getattr(self.graft_backbone, layer_name)
            x = res_layer(x)
        outs.append(x)
        # print('final X', x.shape)
        return tuple(outs)





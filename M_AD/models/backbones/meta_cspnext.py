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

@MODELS.register_module()
class MetaCSPNeXt(CSPNeXt):
    """CSPNeXt backbone used in RTMDet.
    Meta-ResNet used in Meta-RCNN
    """
    def __init__(self,
                 arch,
                 widen_factor,
                 norm_cfg,
                 act_cfg,
                 *args,
                 **kwargs) -> None:
        super().__init__(arch=arch,
                         widen_factor=widen_factor,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         *args,
                         **kwargs)
        arch_setting = self.arch_settings[arch]

        # ----- meta_stem with input 4 dimension
        self.meta_stem = nn.Sequential(
            ConvModule(
                4,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x: Tuple[Tensor, ...], use_meta_conv=False) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            # --- use meta stem
            if layer_name == 'stem' and use_meta_conv:
                layer = self.meta_stem
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)





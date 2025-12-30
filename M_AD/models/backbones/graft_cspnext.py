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
from copy import deepcopy
@MODELS.register_module()
class GraftCSPNeXt(CSPNeXt):
    """CSPNeXt backbone used in RTMDet.
    Meta-ResNet used in Meta-RCNN
    """
    def __init__(self,
                 arch,
                 widen_factor,
                 norm_cfg,
                 act_cfg,
                 after_layer=2,
                 *args,
                 **kwargs) -> None:
        super().__init__(arch=arch,
                         widen_factor=widen_factor,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         *args,
                         **kwargs)
        arch_setting = self.arch_settings[arch]

        self.after_layer = after_layer
        self.ex_init = False
        for i, layer_name in enumerate(self.layers):
            if i < self.after_layer:
                continue
            layer = getattr(self, layer_name)
            self.add_module(f'ex_{layer_name}', deepcopy(layer))

    def init_ex_layers(self):
        if not self.ex_init:
            for i, layer_name in enumerate(self.layers):
                if i < self.after_layer:
                    continue
                layer = getattr(self, layer_name)
                ex_layer = getattr(self, f'ex_{layer_name}')
                ex_layer.load_state_dict(layer.state_dict())
                print('Ex layer load from:', layer.state_dict().keys())
            self.ex_init = True

    def forward_embed(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        self.init_ex_layers()
        outs = []
        for i, layer_name in enumerate(self.layers):
            if i < self.after_layer:
                pass
            else:
                layer_name = f'ex_{layer_name}'
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)





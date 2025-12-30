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
import torch
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.se_layer import ChannelAttention
from mmdet.models.layers.csp_layer import DarknetBottleneck, CSPNeXtBlock, CSPLayer
from mmdet.models.backbones.csp_darknet import SPPBottleneck

class MoESequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward_embed(self, input: Tensor) -> Tensor:
        for module in self:
            if hasattr(module, 'forward_embed'):
                input = module.forward_embed(input)
            else:
                input = module(input)
        return input
class MoEDarknetBottleneck(DarknetBottleneck):
    def __init__(self,
                 *args,
                 multi_conv1=False,
                 multi_conv2=False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multi_conv1 = multi_conv1
        self.multi_conv2 = multi_conv2
        if self.multi_conv1:
            self.ex_conv1 = deepcopy(self.conv1)
        if self.multi_conv2:
            self.ex_conv2 = deepcopy(self.conv2)
        self.ex_init = False

    def ex_init_weights(self):
        if not self.ex_init:
            if self.multi_conv1:
                self.ex_conv1.load_state_dict(self.conv1.state_dict())
                print('EX_CONV1 load from:', self.conv1.state_dict().keys())
            if self.multi_conv2:
                self.ex_conv2.load_state_dict(self.conv2.state_dict())
                print('EX_CONV2 load from:', self.conv2.state_dict().keys())
            self.ex_init = True

    def forward_embed(self, x):
        self.ex_init_weights()
        identity = x
        # print(torch.sum(self.ex_conv1.conv.weight != self.conv1.conv.weight))
        out = self.conv1(x) if not self.multi_conv1 else self.ex_conv1(x)
        out = self.conv2(out) if not self.multi_conv2 else self.ex_conv2(out)
        if self.add_identity:
            return out + identity
        else:
            return out


class MoECSPNeXtBlock(CSPNeXtBlock):

    def __init__(self,
                 *args,
                 multi_conv1=False,
                 multi_conv2=False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multi_conv1 = multi_conv1
        self.multi_conv2 = multi_conv2
        if self.multi_conv1:
            self.ex_conv1 = deepcopy(self.conv1)
        if self.multi_conv2:
            self.ex_conv2 = deepcopy(self.conv2)
        self.ex_init = False

    def ex_init_weights(self):
        if not self.ex_init:
            if self.multi_conv1:
                self.ex_conv1.load_state_dict(self.conv1.state_dict())
                print('EX_CONV1 load from:', self.conv1.state_dict().keys())
            if self.multi_conv2:
                self.ex_conv2.load_state_dict(self.conv2.state_dict())
                print('EX_CONV2 load from:', self.conv2.state_dict().keys())
            self.ex_init = True

    def forward_embed(self, x):
        self.ex_init_weights()
        identity = x
        # print(torch.sum(self.ex_conv1.conv.weight != self.conv1.conv.weight))
        out = self.conv1(x) if not self.multi_conv1 else self.ex_conv1(x)
        out = self.conv2(out) if not self.multi_conv2 else self.ex_conv2(out)
        if self.add_identity:
            return out + identity
        else:
            return out


class MoECSPLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 #####
                 multi_conv1=False,
                 multi_conv2=False,
                 #####
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        block = MoECSPNeXtBlock if use_cspnext_block else MoEDarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = MoESequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                multi_conv1=multi_conv1,
                multi_conv2=multi_conv2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

    def forward_embed(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        #########################
        # for i in range(len(self.blocks)):
        #     x_main = self.blocks[i].forward_embed(x_main)
        x_main = self.blocks.forward_embed(x_main)
        #########################

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

@MODELS.register_module()
class MoECSPNeXt(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
            self,
            arch: str = 'P5',
            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            out_indices: Sequence[int] = (2, 3, 4),
            frozen_stages: int = -1,
            use_depthwise: bool = False,
            expand_ratio: float = 0.5,
            arch_ovewrite: dict = None,
            spp_kernel_sizes: Sequence[int] = (5, 9, 13),
            channel_attention: bool = True,
            #####
            multi_conv1=False,
            multi_conv2=False,
            #####
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU'),
            norm_eval: bool = False,
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.stem = MoESequential(
            ConvModule(
                3,
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
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = MoECSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                #####
                multi_conv1=multi_conv1,
                multi_conv2=multi_conv2,
                #####
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', MoESequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward_embed(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer.forward_embed(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

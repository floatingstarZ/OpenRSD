# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmrotate.registry import MODELS
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN

@MODELS.register_module()
class CSPNeXtPAFPNAdaptChannels(CSPNeXtPAFPN):
    def __init__(self, real_in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.real_in_channels = real_in_channels
        self.in_convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.in_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.real_in_channels[i],
                              self.in_channels[i],
                              1, 1, 0)
                )
            )


    def forward(self, org_inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(org_inputs) == len(self.in_channels)
        inputs = []
        for idx, conv in enumerate(self.in_convs):
            inputs.append(conv(org_inputs[idx]))

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)

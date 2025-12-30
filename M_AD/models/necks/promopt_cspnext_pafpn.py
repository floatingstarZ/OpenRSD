# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

# from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.layers import CSPLayer
from mmrotate.registry import MODELS
from mmdet.models.necks import CSPNeXtPAFPN
import math
from typing import Tuple, Type
from M_AD.models.utils.transformer_modular import TwoWayTransformerModularYOLO

@MODELS.register_module()
class PromptCSPNeXtPAFPN(CSPNeXtPAFPN):
    def __init__(
        self,*args,
            conv_cfg: bool = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='Swish'),
            **kwargs,
    ) -> None:
        super().__init__(
            *args, conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,**kwargs)

        self.cross_in_channels = [192, 192, 384]
        self.cross_in_convs = nn.ModuleList()
        self.cross_out_convs = nn.ModuleList()
        embed_dims = max(self.cross_in_channels)
        for channels in self.cross_in_channels:
            in_convs = ConvModule(
                    channels,
                    embed_dims,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            out_convs = ConvModule(
                    embed_dims,
                    channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.cross_in_convs.append(in_convs)
            self.cross_out_convs.append(out_convs)
        self.cross_attention = TwoWayTransformerModularYOLO(embedding_dim=embed_dims,
                                                            num_heads=8,
                                                            mlp_dim=1024,
                                                            with_query_self_attn=False,
                                                            with_cross_query_to_key=True,
                                                            with_cross_key_to_query=True,
                                                            depth=3)

    def forward_in_context(self, inputs, support_feats):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

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

        ###############################
        spatial_shapes = [f.shape[2:] for f in inner_outs]
        cross_in_feats = [self.cross_in_convs[i](f) for i, f in enumerate(inner_outs)]
        # ---- [B C H W] -> B N C
        img_feats = torch.cat([f.flatten(2) for f in cross_in_feats], dim=-1).permute(0, 2, 1)
        B, N, C = img_feats.shape
        M, _ = support_feats.shape
        support_feats = support_feats[None, ...].expand([B, M, C])
        out_support_feats, out_img_feats = self.cross_attention(support_feats, img_feats)
        # ---- B N C -> [B C H W]
        out_img_feats = out_img_feats.permute(0, 2, 1)
        cross_out_feats = []
        start = 0
        for shape in spatial_shapes:
            H, W = shape
            cross_out_feats.append(out_img_feats[:, start: start + H*W, :].reshape(B, C, H, W).contiguous())
        inner_outs = [self.cross_out_convs[i](f) for i, f in enumerate(cross_out_feats)]
        ###############################

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

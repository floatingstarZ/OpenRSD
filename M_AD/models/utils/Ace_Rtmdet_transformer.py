# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torchvision.ops import boxes

from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.registry import MODELS

from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import List, Tuple, Union
from torch import Tensor

from mmengine.structures.instance_data import InstanceData
from copy import deepcopy
from ctlib.os import *
import torch.nn.functional as F
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmdet.structures.bbox import bbox2roi
import mmdet.registry as mmd_registry
from mmrotate.structures.bbox import rbbox_overlaps
from mmdet.models.detectors import RTMDet
import math
from mmengine.model import xavier_init, normal_init
from mmengine.model import BaseModule, ModuleList
from copy import deepcopy
from mmdet.models.layers.transformer.detr_layers import \
    DetrTransformerDecoder, DetrTransformerDecoderLayer, DetrTransformerEncoderLayer
from mmdet.models.layers.transformer.utils import \
    MLP, ConditionalAttention, coordinate_to_encoding, inverse_sigmoid


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@MODELS.register_module()
class AceRtmdet_transformer(BaseModule):
    """
    v8效果不好，考虑引入LoRA、Grafting来让两个网络区分性更大
    """

    def __init__(self,
                 decoder,
                 num_feature_levels=5,
                 two_stage_num_proposals=300,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.decoder = DNARSDeformableDetrTransformerDecoder(**decoder)
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dims)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_init(self.level_embeds)
        nn.init.normal_(self.query_embed.weight.data)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_transformer(self,
                            #####
                            output_proposals,
                            #####
                            mlvl_feats,
                            mlvl_masks,
                            query_embed,
                            mlvl_pos_embeds,
                            dn_label_query,
                            dn_bbox_query,
                            bbox_coder=None,
                            reg_branches=None,
                            cls_branches=None,
                            angle_braches=None,
                            angle_coder=None,
                            attn_masks=None,
                            **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            # pos_embed.shape = [2, 256, 128, 128]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # [bs, w*h, c]
            feat = feat.flatten(2).transpose(1, 2)
            # [bs, w*h]
            mask = mask.flatten(1)
            # [bs, w*h]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # multi-scale reference points

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        # ---------- memory: B N(num Key) 256
        memory = feat_flatten.permute(1, 0, 2)
        bs, _, c = memory.shape
        # ---------- output_memory:    B N(num Key) 256，将memory经过了一层变换
        # ---------- output_proposals: B N(num Key) 4，  Proposal还是和Anchor一样，是生成的
        output_memory = memory
        # ---------- enc_outputs_class:          B N(num Key) 15，   将memory经过了一层变换
        # ---------- enc_outputs_angle_cls:      B N(num Key) 180，  ARS_CSL的角度编码输出
        # ---------- enc_outputs_coord_unact:    B N(num Key) 4，    坐标输出
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_angle_cls = angle_braches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = \
            reg_branches[self.decoder.num_layers](output_memory) + output_proposals
        # ---------- 选取topk个，enc_outputs_class实际上只有第一个值有意义（Proposal前背景分类）
        # 因此只使用了第0维进行筛选。最后对angle进行解码，获得topk_angle
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        topk_angle_cls = torch.gather(
            enc_outputs_angle_cls, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_angle_cls.shape[-1])
        )
        topk_angle = angle_coder.decode(topk_angle_cls.detach()).detach()
        # ---------- 将query与dnquery（denoise）联合起来，准备decoder工作
        query = self.query_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query[..., :4], topk_coords_unact], dim=1)
            reference_angle = torch.cat([dn_bbox_query[..., 4], topk_angle], dim=1)
        else:
            reference_points = topk_coords_unact
            reference_angle = topk_angle
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        init_reference_angle_out = reference_angle

        # ------------- decoder过程
        # INPUT：
        #   reference_points: B N(Query) 4
        #   reference_angle:  B N(Query)    这两个最为主要，记录了坐标与角度
        # Output
        # inter_stater:     N(Decoder Layer)   N(Query) B 256
        # inter_references: N(Decoder Layer+1) B N(Query) 4，
        #           第一个元素为初始的reference point，最后一个维度为4是因为只记录了坐标，而没有角度

        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_masks,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            bbox_coder=bbox_coder,
            reference_angle=reference_angle,
            angle_braches=angle_braches,
            angle_coder=angle_coder,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, init_reference_angle_out, \
            inter_references_out, enc_outputs_class, \
            enc_outputs_coord_unact, enc_outputs_angle_cls


class DNARSDeformableDetrTransformerDecoder(BaseModule):

    def __init__(self,
                 return_intermediate=False,
                 transformerlayers=None,
                 num_layers=None,
                 init_cfg=None):
        super(DNARSDeformableDetrTransformerDecoder, self).__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(DetrTransformerDecoderLayer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        self._init_layers()
        self.return_intermediate = return_intermediate
        self.rotate_deform_attn = False

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        # self.angle_head = build_MLP(self.embed_dims, self.embed_dims,
        #                                 self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000**(2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                bbox_coder=None,
                reference_angle=None,
                angle_braches=None,
                angle_coder=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            reference_angle_input = reference_angle[:, :, None]
            if self.rotate_deform_attn:
                index = (reference_angle_input > math.pi / 4).view(-1) | \
                        (reference_angle_input < -math.pi / 4).view(-1)
                bs, q_num, layer_num, points_num = reference_points_input.shape
                reference_points_input = reference_points_input.view(bs * q_num, layer_num, points_num).contiguous()
                reference_points_input[index, :, 2], reference_points_input[index, :, 3] = \
                    reference_points_input[index, :, 3], reference_points_input[index, :, 2]
                reference_points_input = reference_points_input.view(bs, q_num, layer_num, points_num).contiguous()
                reference_angle_input = (reference_angle_input + math.pi / 4 - reference_angle_input * 1e-10) % \
                                        (math.pi / 2) - math.pi / 4 + reference_angle_input * 1e-10
            kwargs['reference_angles'] = reference_angle_input
            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if angle_braches is not None:
                reference_angle = angle_coder.decode(angle_braches[lid](output)).detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points



def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    # TODO: It can be implemented by add an out_channel arg of
    #  mmcv.cnn.bricks.transformer.FFN
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)








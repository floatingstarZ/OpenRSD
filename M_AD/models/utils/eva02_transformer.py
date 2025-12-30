import torch
import torch.nn.functional as F
import torch.nn as nn

# ---- from mmpretrain.models.backbones.vit_eva02.py
from typing import Union, Optional
import numpy as np
from mmcv.cnn.bricks.drop import build_dropout

class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self,
                 dim: int,
                 layer_scale_init_value: Union[float, torch.Tensor] = 1e-5,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
        dropout_layer: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)

        if norm_cfg is not None:
            self.norm = nn.LayerNorm(hidden_dims)
        else:
            self.norm = nn.Identity()

        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self,
                x: torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            # due to the dimension inconsistence or user setting
            # not to apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out

class AttentionEva02(nn.Module):
    """Multi-head Attention Module with 2D sincos position embedding (RoPE).

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q and v. Note
            that we follows the official implementation where ``k_bias``
            is 0. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 init_cfg=None):
        super(AttentionEva02, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EVA02EndcoderLayer(nn.Module):
    """Implements one encoder EVA02EndcoderLayer in EVA02.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension of FFNs.
        sub_ln (bool): Whether to add the sub layer normalization
            in the attention module. Defaults to False.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool): enable bias for projection in the attention module
            if True. Defaults to True.
        rope (:obj:`torch.nn.Module`, optional): RotaryEmbedding object
            in the attention module. Defaults to None.
        drop_rate (float): Dropout rate in the mlp module. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_cfg=None):
        super(EVA02EndcoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.attn = AttentionEva02(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        self.norm2 = nn.LayerNorm(embed_dims)

        if drop_rate > 0:
            dropout_layer = dict(type='Dropout', drop_prob=drop_rate)
        else:
            dropout_layer = None

        self.mlp = SwiGLUFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            dropout_layer=dropout_layer,
            norm_cfg=None,
            add_identity=False,
        )

    def forward(self, x):
        inputs = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = inputs + x

        inputs = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = inputs + x

        return x

class EVA02Decoder(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_layers=2,
                 init_cfg=None):
        super(EVA02Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EVA02EndcoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    proj_bias=proj_bias,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate
                )
            )
        self.out_fc = nn.Linear(embed_dims, 1)

    def forward(self, x):
        # B x N x 256
        for layer in self.layers:
            x = layer(x)
        # B x N x 1
        x = self.out_fc(x)
        return x

class EVA02Decoderv2(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_layers=2,
                 init_cfg=None):
        super(EVA02Decoderv2, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EVA02EndcoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    proj_bias=proj_bias,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate
                )
            )


    def forward(self, x):
        # B x N x 256
        for layer in self.layers:
            x = layer(x)
        # B x N x 256
        return x

if __name__ == '__main__':
    ####--- test 1
    # decoder = EVA02EndcoderLayer(256, 256, 256)
    # pred_feats = torch.rand([30, 16, 256])
    # print(decoder(pred_feats).shape)

    ####--- test 2
    decoder = EVA02Decoder(embed_dims=256, num_heads=8, feedforward_channels=256, num_layers=2)
    pred_feats = torch.rand([30, 256])
    support_feats = torch.rand([16, 256])

    N = pred_feats.shape[0]
    pred_feats = pred_feats[:, None, :] # 30 x 1 x 256
    support_feats = support_feats[None, ...].expand(N, -1, -1) # 30 x 16 x 256
    decode_feats = torch.cat([pred_feats, support_feats], dim=1)
    print(decoder(decode_feats).shape)


















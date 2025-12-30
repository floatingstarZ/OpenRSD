# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Modified from SAM Decoder
"""
import torch
from torch import Tensor, nn
import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Type

from .common import MLPBlock
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoder

# Define the softmax_one function with added one in the denominator , which helps to reduce
#the negative impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class PureMLP(nn.Module):
    def __init__(
            self,
            depth=2,
            embedding_dim=256,
            num_heads=8,
            mlp_dim=2048,
            activation=nn.ReLU,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                MLPLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation
                )
            )

    def forward(self, queries):
        """

        :param image_embedding:  B x N x C，图像Encoder的特征
        :param query_embeddings: B x N x C，DETR Encoder 的特征
        image -> selfattention -> cross attention to get detr feature -> cross to detr feature -> output
        2 layers
        :return:
        """

        # Apply transformer blocks and final layernorm
        # 两次双向特征交互
        for layer in self.layers:
            queries = layer(queries=queries)
        return queries


class TwoWayTransformerModularYOLO(nn.Module):
    def __init__(
            self,
            depth=2,
            embedding_dim=256,
            num_heads=8,
            mlp_dim=2048,
            activation=nn.ReLU,
            # ---- 一些定制化参数
            # -- 主体是query，query_to_key代表key也会进行更新
            with_query_self_attn=False,
            with_cross_query_to_key=False,
            with_cross_key_to_query=False,
            with_key_self_attn=False,
            # ---- 是否使用softmax1
            use_softmax1=False
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    with_query_self_attn=with_query_self_attn,
                    with_cross_query_to_key=with_cross_query_to_key,
                    with_cross_key_to_query=with_cross_key_to_query,
                    with_key_self_attn=with_key_self_attn,
                    use_softmax1=use_softmax1
                )
            )
        if use_softmax1:
            print('use_softmax1 ' * 100)

    def forward(
            self,
            queries: Tensor,
            keys: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """

        :param image_embedding:  B x N x C，图像Encoder的特征
        :param query_embeddings: B x N x C，DETR Encoder 的特征
        image -> selfattention -> cross attention to get detr feature -> cross to detr feature -> output
        2 layers
        :return:
        """

        # Apply transformer blocks and final layernorm
        # 两次双向特征交互
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
            )
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            with_query_self_attn=False,
            with_cross_query_to_key=False,
            with_cross_key_to_query=False,
            with_key_self_attn=False,
            use_softmax1=False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.with_query_self_attn = with_query_self_attn
        self.with_cross_query_to_key = with_cross_query_to_key
        self.with_cross_key_to_query = with_cross_key_to_query
        self.with_key_self_attn = with_key_self_attn

        # ---- query是主体，但可以没有自己的attention
        if self.with_query_self_attn:
            self.self_attn = Attention(embedding_dim, num_heads, use_softmax1=use_softmax1)
            self.norm1 = nn.LayerNorm(embedding_dim)
        if self.with_key_self_attn:
            self.self_attn_k = Attention(embedding_dim, num_heads, use_softmax1=False)
            self.norm1_k = nn.LayerNorm(embedding_dim)

        # ---- query是主体，因此key_to_query是必须的，query一定要获得key的特征
        if self.with_cross_key_to_query:
            self.cross_attn_token_to_image = Attention(
                embedding_dim, num_heads, use_softmax1=use_softmax1)
            self.norm2 = nn.LayerNorm(embedding_dim)
        # ---- 一层MLP
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # ---- query是主体，key可以不获得query的特征
        if self.with_cross_query_to_key:
            # self.norm4 = nn.Sequential(
            #     nn.Linear(embedding_dim, embedding_dim),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(embedding_dim, embedding_dim)
            # )
            self.norm4 = nn.LayerNorm(embedding_dim)

            self.cross_attn_image_to_token = Attention(
                embedding_dim, num_heads, use_softmax1=use_softmax1)
        self.linear_k = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.norm_k = nn.LayerNorm(embedding_dim)

    def forward(
            self, queries: Tensor, keys: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """

        :param queries:  support_feats
        :param keys:     in_flatten_feats
        :return:
        """
        # query的Self attention block
        if self.with_query_self_attn:
            q = queries
            attn_out = self.self_attn(q=q, k=q, v=q)
            queries = queries + attn_out
            queries = self.norm1(queries)

        # query的Self attention block
        if self.with_key_self_attn:
            q = keys
            attn_out = self.self_attn_k(q=q, k=q, v=q)
            keys = keys + attn_out
            keys = self.norm1_k(keys)

        # Cross attention block, query获得key的特征
        if self.with_cross_key_to_query:
            q = queries
            k = keys
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out
            queries = self.norm2(queries)

        # MLP block
        # if self.with_query_self_attn:
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # key同样可以获得query的特征
        if self.with_cross_query_to_key:
            q = queries
            k = keys
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
            keys = keys + attn_out
            keys = self.norm4(keys)

        keys = self.linear_k(keys)
        keys = self.norm_k(keys)

        return queries, keys



class SupportAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            with_cross_query_to_key=False,
            with_cross_key_to_query=True,
            use_softmax1=False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.with_cross_query_to_key = with_cross_query_to_key
        self.with_cross_key_to_query = with_cross_key_to_query

        # ---- query是主体，因此key_to_query是必须的，query一定要获得key的特征
        if self.with_cross_key_to_query:
            self.cross_attn_token_to_image = Attention(
                embedding_dim, num_heads, use_softmax1=use_softmax1)
            self.norm2 = nn.LayerNorm(embedding_dim)
        # ---- 一层MLP
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # ---- query是主体，key可以不获得query的特征
        if self.with_cross_query_to_key:
            self.norm4 = nn.LayerNorm(embedding_dim)
            self.cross_attn_image_to_token = Attention(
                embedding_dim, num_heads, use_softmax1=use_softmax1)

    def forward(
            self, queries: Tensor, keys: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """

        :param queries:  support_feats
        :param keys:     memories
        :return:
        """

        # Cross attention block, query获得memory的特征
        if self.with_cross_key_to_query:
            q = queries
            k = keys
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out
            queries = self.norm2(queries)

        # MLP block
        # if self.with_query_self_attn:
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # key同样可以获得query的特征
        if self.with_cross_query_to_key:
            q = queries
            k = keys
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
            keys = keys + attn_out
            keys = self.norm4(keys)

        return queries, keys


# 就是很普通的Attention模块，qkv分别进行映射，Attention，然后输出有个映射
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            use_softmax1=False
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.use_softmax1 = use_softmax1
        if self.use_softmax1:
            self.softmax = softmax_one
        else:
            self.softmax = torch.softmax


    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = self.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLPLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU
    ) -> None:
        super().__init__()
        # ---- 一层MLP
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)


    def forward(self, queries):

        # MLP block
        # if self.with_query_self_attn:
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries
import torch
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast
from einops import rearrange, pack, unpack

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class FSQAutoEncoder_with_loss(Module):
    def __init__(self,
                 embed_dims=768,
                 qua_type='V1',
                 qua_code_size=1024):
        super().__init__()
        self.rec_loss = torch.nn.L1Loss()
        assert qua_code_size in [1024, 4096, 8192]
        if qua_code_size == 1024:
            levels = [4, 4, 4, 4, 2, 2]
        elif qua_code_size == 4096:
            levels = [4, 4, 4, 4, 4, 4]
        elif qua_code_size == 8192:
            levels = [4, 4, 4, 4, 4, 4, 2]

        # ----- 量化只用来解析语义，然而检测输出的分类分数还需要包含置信度这一连续信息，亦或者是IoU信息
        # ----- 因此这里添加一个side_branch来补充这部分信息
        self.side_branch = nn.Sequential(
            nn.Linear(embed_dims, 2),
            nn.Linear(2, embed_dims)
        )
        if qua_type in ['V1']:
            """
            quantizer的Encoder具有更强的非线性能力，编码数量也更多
            """
            # ----- 量化器，输出量化特征。1024 codes in codebook,
            self.quantizer = FSQ(levels=levels, dim=embed_dims)
            # ----- 编解码器，目的就是增加一些非线性性。Decoder只用来计算损失
            self.encoder = nn.Sequential(
                nn.Linear(embed_dims, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, embed_dims)
            )
            self.decoder = nn.Identity()
        elif qua_type in ['V2']:
            """
            假设视觉Embeddings已经是在一个空间内了，使用Bottle Neck + Qua来压缩信息
            Qua的数量限制在1024，并且维度限制在256
            """
            # ----- 编解码器，目的就是增加一些非线性性。Decoder只用来计算损失
            self.encoder = nn.Sequential(
                nn.Linear(embed_dims, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
            )
            # ----- 量化器，输出量化特征。1024 codes in codebook,
            self.quantizer = FSQ(levels=levels, dim=256)
            self.decoder = nn.Sequential(
                nn.Linear(256, embed_dims),
            )
        elif qua_type in ['V3']:
            """
            不使用真正的量化，只进行压缩
            """
            # ----- 编解码器单纯的压缩到256维特征，并且经过ReLu进一步缩减特征空间大小
            self.encoder = nn.Sequential(
                nn.Linear(embed_dims, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                VecNorm(),
                nn.Linear(256, embed_dims)
            )
            self.quantizer = nn.Identity()
            self.decoder = nn.Identity()
        elif qua_type in ['V4']:
            """
            使用更多Levels
            """
            if qua_code_size == 1024:   # 1920
                levels = [8, 8, 6, 5]
            elif qua_code_size == 4096: # 15360
                levels = [8, 8, 8, 6, 5]
            elif qua_code_size == 8192: # 15360
                levels = [8, 8, 8, 5, 5, 5]
            # ----- 编解码器，目的就是增加一些非线性性。Decoder只用来计算损失
            self.encoder = nn.Sequential(
                nn.Linear(embed_dims, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
            )
            # ----- 量化器，输出量化特征。1024 codes in codebook,
            self.quantizer = FSQ(levels=levels, dim=256)
            self.decoder = nn.Sequential(
                nn.Linear(256, embed_dims),
            )
        else:
            raise Exception('Unsupported quantization type,', qua_type)

    def forward(self, x):
        """

        :param x: N x C
        :return:
        """
        l = self.encoder(x)
        side_info = self.side_branch(x)
        quantized_l = self.quantizer(l)
        x_rec = self.decoder(quantized_l) * side_info
        return x_rec, torch.tensor(0.0).to(x.device), None

class VecNorm(Module):
    def __init__(self):
        super(VecNorm, self).__init__()
    def forward(self, x):
        normed = torch.nn.functional.normalize(x, dim=-1)
        return normed

class VAE(Module):
    """
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """
    def __init__(self, embed_dims=256):
        super(VAE, self).__init__()
        self.fc_mu = nn.Linear(embed_dims, embed_dims)
        self.fc_logvar = nn.Linear(embed_dims, embed_dims)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu = self.fc_mu(input)
        log_var = self.fc_logvar(input)
        z = self.reparameterize(mu, log_var)
        return  z


class FSQAutoEncoder_with_loss_VER2(Module):
    def __init__(self, embed_dims=768, hidden_dim=256):
        super().__init__()
        # ----- 量化器，输出量化特征。4096=4^6 codes in codebook,
        self.levels = [4, 4, 4, 4, 4, 4]
        hidden_dim = len(self.levels)
        mlp_dim = 1024
        self.rec_loss = torch.nn.L1Loss()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dims, mlp_dim),
            nn.ReLU(True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(True),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.quantizer = FSQ(levels=self.levels)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, embed_dims)
        )

    def forward(self, x):
        """

        :param x: N x C
        :return:
        """
        l = self.encoder(x)
        quantized_l, code_indices = self.quantizer(l)
        x_rec = self.decoder(quantized_l)
        rec_loss = self.rec_loss(x, x_rec)
        return x_rec, rec_loss, code_indices

class FSQ(Module):
    def __init__(
            self,
            levels: List[int],
            dim = None,
            num_codebooks=1,
            keep_num_codebooks_dim = None,
            scale: float = None,
            allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
            channel_first: bool = False,
            projection_has_bias: bool = True
    ):
        super().__init__()
        # ---- Latent embeddings的维度（len(_levels)）和取值范围
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)
        # ---- 编码偏置（基）
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        # ---- 暂时没用
        self.scale = scale

        # ---- 也是隐层特征的维度
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        # ---- 应该是可以多个（num_codebooks）codebook，但暂时不用
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # ---- 输入特征维度dim，判断是否需要进行映射
        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim,
                                    bias=projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim,
                                     bias=projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        # ---- codeboox的数量为所有levels的乘积（其实很大）
        self.codebook_size = self._levels.prod().item()

        # ---- N_code x len(level)，每个元素为code对应的隐式编码
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes

    def bound(self, z, eps: float = 1e-3):
        # --- 限制范围
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        # --- 量化
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled=False)
    def forward(self, z):
        """

        :param z: (b, n, d, c)
            einstein notation
            b - batch
            n - sequence (or flattened spatial dimensions)
            d - feature dimension
            c - number of codebook dim (实际应该是codebook的数量吧)
        :return:
            out: (b, n, d, c)
            indices
        """


        orig_dtype = z.dtype
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # make sure allowed dtype before quantizing

        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        # ---- 进行量化，以及量化对应的code的indices
        codes = self.quantize(z)
        # indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # --- cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # ---- project out，映射回到连续特征

        out = self.project_out(codes)

        # ----reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            # indices = unpack_one(indices, ps, 'b * c')

        # if not self.keep_num_codebooks_dim:
        #     indices = rearrange(indices, '... 1 -> ...')

        # return quantized output and indices

        return out # , indices


if __name__ == '__main__':
    print('ORG Test')
    quantizer = FSQ(
        levels = [8, 5, 5, 5]
    )

    x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    # (1, 1024, 4), (1, 1024)

    assert torch.all(xhat == quantizer.indices_to_codes(indices))
    print(xhat.shape, indices.shape)
    print(xhat, indices)
    ####################################
    print('Test in MMRotate_AD')
    quantizer = FSQ(
        levels=[4, 4, 4, 4],
        dim=768
    )

    x = torch.randn(1, 10, 768)  # B x N x C
    xhat, indices = quantizer(x)

    # (1, 1024, 4), (1, 1024)

    assert torch.all(xhat == quantizer.indices_to_codes(indices))
    print(xhat.shape, indices.shape)
    print(xhat, indices)

    ##############################
    print('Test in FSQAutoEncoder_with_loss')
    quantizer = FSQAutoEncoder_with_loss(embed_dims=768)
    x = torch.randn(1, 15, 768)  # B x N x C
    x_rec, rec_loss, code_indices = quantizer(x)
    print(x_rec.shape, code_indices.shape)
    print(x_rec, rec_loss, code_indices)
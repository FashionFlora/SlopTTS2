# coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import  spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from collections import OrderedDict
from Modules.diffusion.dit_style_net import DiTStyleNet
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Modules.diffusion.diffusion import AudioDiffusionConditional
from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution , VKDiffusion
from Modules.diffusion.style_prior import StyleDiT, KDiffusionWithEMA
from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
from Modules.diffusion.dit1d_copy import Transformer1d
from munch import Munch
import yaml


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)



class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 8, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock(nn.Module):
    """Residual block with optional downsample and SE. Uses GroupNorm(1)."""
    def __init__(self, in_ch, out_ch, downsample=False,
                 use_gn=True, use_se=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.use_se = use_se
        self.norm1 = nn.GroupNorm(1, in_ch) if use_gn else nn.Identity()
        self.conv1 = weight_norm(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.norm2 = nn.GroupNorm(1, in_ch) if use_gn else nn.Identity()
        self.conv2 = weight_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
                      bias=False)
        )
        if in_ch != out_ch or stride != 1:
            self.proj = weight_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                          padding=0, bias=False)
            )
        if use_se:
            self.se = SEBlock(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        idt = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        if hasattr(self, "proj"):
            idt = self.proj(idt)
        return (idt + out) / math.sqrt(2)


class SpatialSelfAttention(nn.Module):
    """MultiHead attention over spatial positions (H*W)."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by heads"
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        seq = H * W
        x_flat = x.view(B, C, seq).permute(0, 2, 1)  # (B, seq, C)
        y, _ = self.mha(x_flat, x_flat, x_flat)
        y = self.proj(y)
        y = self.norm(y + x_flat)
        y = y.permute(0, 2, 1).view(B, C, H, W)
        return y


class StyleEncoderGAN(nn.Module):
    """
    Encoder for vocoder (GAN) usage.
    - Input: (B, 1, time, n_mels)
    - Output: (B, style_dim)
    Defaults: dim_in=64, style_dim=128, repeat=5, max_conv_dim=512.
    """
    def __init__(self, dim_in=64, style_dim=128, repeat=5,
                 max_conv_dim=512, use_se=True, use_attn=True,
                 attn_heads=8, pool_last_k=3, use_weight_norm=True):
        super().__init__()
        assert dim_in > 0 and style_dim > 0
        self.pool_last_k = min(pool_last_k, repeat)
        self.use_attn = use_attn
        wn = weight_norm if use_weight_norm else (lambda x: x)

        self.stem = nn.Sequential(
            wn(nn.Conv2d(1, dim_in, kernel_size=3, stride=1, padding=1,
                         bias=False)),
            nn.GroupNorm(1, dim_in),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.stages = nn.ModuleList()
        channels = []
        cur = dim_in
        for _ in range(repeat):
            nxt = min(cur * 2, max_conv_dim)
            self.stages.append(
                ResBlock(cur, nxt, downsample=True, use_gn=True, use_se=use_se)
            )
            channels.append(nxt)
            cur = nxt

        if use_attn:
            self.attn = SpatialSelfAttention(cur, num_heads=attn_heads)

        pooled_dim = sum(channels[-self.pool_last_k:])
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(pooled_dim // 2, style_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B,1,time,n_mels) -> s: (B, style_dim)
        collects global‑avg from each stage and concatenates last K pools.
        """
        feats = []
        h = self.stem(x)
        for idx, stage in enumerate(self.stages):
            h = stage(h)
            # apply attention to final stage (bottleneck) if enabled
            if idx == len(self.stages) - 1 and self.use_attn:
                h = self.attn(h)
            pooled = F.adaptive_avg_pool2d(h, 1).view(h.size(0), -1)
            feats.append(pooled)

        pooled = torch.cat(feats[-self.pool_last_k:], dim=1)
        s = self.head(pooled)
        return s
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = dim
        half = (dim + 1) // 2
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        return self.pe[:, :T, :].to(x.device).type_as(x)


def _relative_position_bucket(relative_position,
                              num_buckets=32,
                              max_distance=128,
                              bidirectional=True):
    """
    T5-style bucketing. relative_position is j - i (shape [T, T]).
    Returns long tensor in [0, num_buckets).
    """
    rp = relative_position
    n = rp.abs()
    if bidirectional:
        num_buckets = int(num_buckets)
        half = num_buckets // 2
        # sign bit: positive (future) mapped to upper half
        sign = (rp > 0).long()
        bucket = sign * half
    else:
        # only non-positive relative positions
        bucket = torch.zeros_like(rp, dtype=torch.long)
        n = (-rp).clamp(min=0)

    max_exact = max(1, half // 2)
    is_small = n < max_exact
    # compute logarithmic bins for large distances
    val = max_exact + (
        ( (n.float() / float(max_exact)).clamp(min=1e-6).log() /
          math.log(float(max_distance) / float(max_exact)) ) *
        (half - max_exact)
    ).long()
    val = val.clamp(max=half - 1)
    bucket = bucket + torch.where(is_small, n.long(), val)
    return bucket.long().clamp(max=num_buckets - 1)

class MHSA(nn.Module):
    def __init__(self, hidden: int, n_heads: int = 8, dropout: float = 0.1,
                 num_rel_buckets: int = 32, max_rel_distance: int = 128):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_head = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        self.num_rel_buckets = int(num_rel_buckets)
        self.max_rel_distance = int(max_rel_distance)
        self.rel_bias = nn.Embedding(self.num_rel_buckets, self.n_heads)
        nn.init.normal_(self.rel_bias.weight, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None,
                local_attn_mask: torch.Tensor = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]

        idx = torch.arange(T, device=x.device)
        rel_pos = idx.view(1, -1) - idx.view(-1, 1)  # j - i
        rel_bucket = _relative_position_bucket(rel_pos,
                                               num_buckets=self.num_rel_buckets,
                                               max_distance=self.max_rel_distance)
        # rel_bias(rel_bucket) -> [T, T, H], permute -> [H, T, T]
        rel = self.rel_bias(rel_bucket).permute(2, 0, 1).unsqueeze(0)  # [1,H,T,T]
        scores = scores + rel.to(dtype=scores.dtype, device=scores.device)

        # mask padded keys (True == PAD)
        if key_padding_mask is not None:
            neg = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            scores = scores.masked_fill(mask, neg)

        if local_attn_mask is not None:
            neg = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores + (local_attn_mask.unsqueeze(1).to(dtype=scores.dtype) * neg)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,T,d]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class DurationEncoderConformer(nn.Module):
    """
    Conformer-based DurationEncoder.
    This module processes text embeddings from the main text encoder, conditioned
    on a style vector, to produce prosody-related features.
    """
    def __init__(self, sty_dim, d_model, nlayers, n_heads=4, dropout=0.2, max_pos_len=1024):
        super().__init__()
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_pos_len)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            ConformerBlock(
                style_dim=sty_dim,
                hidden=d_model,
                n_heads=n_heads,
                conv_kernel=15,
                dropout=dropout,
            ) for _ in range(nlayers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x, style, text_lengths, m):
        """
        x: Input text embeddings of shape [B, C, T].
        style: Style vector of shape [B, sty_dim].
        m: Padding mask of shape [B, T] (True where padded).
        """
        h = x.transpose(1, 2)
        
        pos = self.pos_emb(h)
        h = h + pos
        h = self.pos_dropout(h)

        key_padding_mask = m

        for blk in self.blocks:
            h = blk(h, style, key_padding_mask=key_padding_mask, attn_mask=None)

        h = self.final_ln(h)

        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return h
class TextEncoderConformer(nn.Module):
    """
    Conformer-based TextEncoder replacement.

    - Uses token embedding + sinusoidal positional embeddings.
    - Applies a stack of ConformerBlock (existing in your file).
    - Returns [B, C, T] to stay compatible with build_model.
    - forward signature matches previous TextEncoder: forward(x, input_lengths, m)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,  # unused but kept for API compatibility
        depth: int,
        n_symbols: int,
        actv=nn.LeakyReLU(0.2),
        max_pos_len: int = 4096,
        pos_dropout: float = 0.2,
        n_heads: int = 4,
        rel_buckets: int = 32,
        max_rel_distance: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        self.pos_emb = SinusoidalPositionalEmbedding(channels, max_len=max_pos_len)
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.channels = channels

        # build Conformer blocks
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    style_dim=channels,
                    hidden=channels,
                    n_heads=n_heads,
                    conv_kernel=15,
                    dropout=0.2,
                )
                for _ in range(depth)
            ]
        )

        self.final_ln = nn.LayerNorm(channels)

        # store for length_to_mask
        self._max_pos_len = max_pos_len

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor, m: torch.Tensor):
        """
        x: [B, T] token ids
        input_lengths: [B] lengths int
        m: [B, T] mask True=PAD
        returns: [B, C, T]
        """
        B, T = x.shape
        emb = self.embedding(x)  # [B, T, C]
        pos = self.pos_emb(emb)  # [1, T, C]
        h = emb + pos
        h = self.pos_dropout(h)

        # build key_padding_mask expected by ConformerBlock (True=PAD)
        key_padding_mask = m  # [B, T]

        for blk in self.blocks:
            if key_padding_mask is not None:
                # key_padding_mask: True == PAD
                valid = ~key_padding_mask            # True = valid token
                denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1).to(h.dtype)  # [B,1]
                h_masked = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                s = (h_masked.sum(dim=1) / denom)    # [B, C]
            else:
                s = h.mean(dim=1)
            attn_mask = None
            h = blk(h, s, key_padding_mask, attn_mask)

        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        h = self.final_ln(h)  # [B, T, C]
        h = h.transpose(1, 2)  # [B, C, T]
        return h

    def inference(self, x: torch.Tensor):
        emb = self.embedding(x)  # [B, T, C]
        pos = self.pos_emb(emb)
        h = emb + pos
        for blk in self.blocks:
            s = h.mean(dim=1)
            h = blk(h, s, key_padding_mask=None, attn_mask=None)
        h = self.final_ln(h)  # [B, T, C]
        return h.transpose(1, 2)  # [B, C, T]  <-- match forward

    def length_to_mask(self, lengths: torch.Tensor):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
class FFN(nn.Module):
    def __init__(self, hidden: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = hidden * mult
        self.net = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out        
class ConvModule(nn.Module):
    def __init__(self, hidden: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.pw1 = nn.Conv1d(hidden, 2 * hidden, kernel_size=1)
        self.dw = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        # było: self.bn = nn.BatchNorm1d(hidden)
        self.gn = nn.GroupNorm(1, hidden)  # stabilne w eval/train

        self.pw2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)               # [B, C, T]
        x = F.glu(self.pw1(x), dim=1)       # [B, C, T]
        x = self.dw(x)                      # [B, C, T]
        x = self.gn(x)                      # zamiast BN
        x = F.silu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class SALayerNorm(nn.Module):
    def __init__(self, style_dim: int, hidden: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, eps=eps)
        self.affine = nn.Linear(style_dim, hidden * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], s: [B, style_dim]
        y = self.ln(x)
        gamma, beta = self.affine(s).chunk(2, dim=-1)  # [B, C], [B, C]
        gamma = 1.0 + gamma.unsqueeze(1)  # [B, 1, C]
        beta = beta.unsqueeze(1)  # [B, 1, C]
        return gamma * y + beta


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class ConformerBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        hidden: int,
        n_heads: int,
        conv_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.saln_ff1 = SALayerNorm(style_dim, hidden)
        self.ff1 = FFN(hidden, mult=4, dropout=dropout)

        self.saln_mha = SALayerNorm(style_dim, hidden)
        self.mha = MHSA(hidden, n_heads=n_heads, dropout=dropout , num_rel_buckets = 16 ,  max_rel_distance=64)

        self.saln_conv = SALayerNorm(style_dim, hidden)
        self.conv = ConvModule(hidden, kernel_size=conv_kernel, dropout=dropout)

        self.saln_ff2 = SALayerNorm(style_dim, hidden)
        self.ff2 = FFN(hidden, mult=4, dropout=dropout)

        self.final_ln = nn.LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # FFN (1/2)
        x = x + 0.5 * self.ff1(self.saln_ff1(x, s))
        

        # MHSA
        x = x + self.mha(self.saln_mha(x, s), key_padding_mask, attn_mask)

        # Conv
        x = x + self.conv(self.saln_conv(x, s))

        # FFN (2/2)
        x = x + 0.5 * self.ff2(self.saln_ff2(x, s))

        # Final LN
        return self.final_ln(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, rel_bias=None):
        # q: [B, F, C], k,v: [B, T, C]
        B, F, C = q.shape
        T = k.shape[1]

        q = self.q_proj(q).view(B, F, self.n_heads, self.d_head)
        k = self.k_proj(k).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(v).view(B, T, self.n_heads, self.d_head)

        q = q.transpose(1, 2)  # [B, H, F, d]
        k = k.transpose(1, 2)  # [B, H, T, d]
        v = v.transpose(1, 2)  # [B, H, T, d]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: [B, H, F, T]

        if rel_bias is not None:
            # rel_bias: [B, 1, F, T] or [1, 1, F, T]
            scores = scores + rel_bias

        if attn_mask is not None:
            # attn_mask: True where we want to mask/block
            scores = scores.masked_fill(attn_mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, F, d]
        out = out.transpose(1, 2).contiguous().view(B, F, C)  # [B, F, C]
        out = self.out_proj(out)  # [B, F, C]
        return out

def build_monotonic_band_mask(alignment, text_mask, window):
    """
    alignment: [B, T, F] (monotonic hard/soft align)
    text_mask: [B, T] True at padding
    Returns attn_mask: [B, 1, F, T] True where attention is NOT allowed.
    """
    with torch.no_grad():
        B, T, F = alignment.shape
        tau = alignment.argmax(dim=1)  
        t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
        tau_exp = tau.unsqueeze(-1)  
        band = (t_idx >= (tau_exp - window)) & (t_idx <= (tau_exp + window))

        band_mask = ~band  

        # Also mask padded tokens
        key_pad = text_mask.unsqueeze(1).expand(B, F, T) 
        full_mask = band_mask | key_pad  
        return full_mask.unsqueeze(1) 
        
class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim,
        d_hid,
        nlayers,
        max_dur=50,
        dropout=0.1,
        attn_heads=4,
        attn_window=5,
        use_rel_bias=False,
    ):
        super().__init__()

        # używamy Conformerowego enkodera dla duration (zwraca [B, T, d_hid])
        self.text_encoder = DurationEncoderConformer(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            n_heads=attn_heads,
            dropout=dropout,
        )

        # LSTM do predykcji duration: input = d_hid + style_dim
        self.lstm = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

        # F0 / N modules (bez zmian)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout)
        )
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout)
        )
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # cross-attention path
        self.en_channels = d_hid + style_dim  # 512 + 128 = 640
        self.ca_q_norm = AdaLayerNorm(style_dim, self.en_channels)
        self.ca_k_norm = AdaLayerNorm(style_dim, self.en_channels)

        self.cross_attn = CrossAttention(
            d_model=self.en_channels,
            n_heads=attn_heads,
            dropout=dropout,
        )
        self.attn_window = attn_window
        self.use_rel_bias = use_rel_bias

        self.en_post = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    self.en_channels,
                    self.en_channels,
                    kernel_size=5,
                    padding=2,
                    groups=self.en_channels,
                )
            ),
            nn.SiLU(),
            weight_norm(nn.Conv1d(self.en_channels, self.en_channels, kernel_size=1)),
        )

    def compute_en(self, d_tok, alignment, style, text_mask):
        """
        d_tok: [B, T, C] where C == en_channels (d_hid + style_dim)
        alignment: [B, T, F]
        style: [B, S]
        text_mask: [B, T] True at padding
        """
        en_base = torch.matmul(d_tok.transpose(1, 2), alignment)  # [B, C, F]

        q0 = en_base.transpose(1, 2)  # [B, F, C]
        k0 = d_tok  # [B, T, C]

        q = self.ca_q_norm(q0, style)
        k = self.ca_k_norm(k0, style)

        attn_mask = build_monotonic_band_mask(alignment, text_mask, self.attn_window)

        rel_bias = None
        if self.use_rel_bias:
            with torch.no_grad():
                B, T, F = alignment.shape
                tau = alignment.argmax(dim=1)
                t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
                dist = (t_idx - tau.unsqueeze(-1)).abs().float()
                rel_bias = (-0.05 * dist).unsqueeze(1)

        en_attn = self.cross_attn(q, k, k, attn_mask=attn_mask, rel_bias=rel_bias)
        en_attn = en_attn.transpose(1, 2)
        en_attn = self.en_post(en_attn)

        en = (en_base + en_attn) / math.sqrt(2.0)
        return en

    def forward(self, texts, style, text_lengths, alignment, m):
        """
        texts: [B, d_hid, T]  (output from TextEncoderConformer)
        style:  [B, style_dim]
        text_lengths: [B]
        alignment: [B, T, F]
        m: mask [B, T] (True==pad)
        """
        # Duration encoder (Conformer) -> returns [B, T, d_hid]
        d_tok = self.text_encoder(texts, style, text_lengths, m)  # [B, T, d_hid]

        # Concatenate style to each token embedding -> [B, T, d_hid + style_dim]
        style_exp = style.unsqueeze(1).expand(-1, d_tok.size(1), -1)
        d_tok = torch.cat([d_tok, style_exp], dim=-1)

        # Duration head
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d_tok, input_lengths, batch_first=True, enforce_sorted=False
        )

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, : x.shape[1], :] = x
        x = x_pad

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        # compute frame-level prosody encoding
        en = self.compute_en(d_tok, alignment, style, m.squeeze(1))

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask
def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan' , 'ringformer'], 'Decoder type unknown'
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
    elif args.decoder.type == "ringformer":
        from Modules.ringformer import Decoder
        decoder = Decoder(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=args.decoder.gen_istft_n_fft,
            gen_istft_hop_size=args.decoder.gen_istft_hop_size,
        )
        
    text_encoder = TextEncoderConformer(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    
    predictor_encoder = StyleEncoderGAN(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
    style_encoder = StyleEncoderGAN(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    num_layers = 3
    channels = 256               # mel-channel output dim expected by your pipeline
    num_heads = 8
    head_features = 128
    multiplier = 2
    context_embedding_features = 768  # e.g., BERT hidden size
    embedding_max_length = 512    
    dit = Transformer1d(
        num_layers=num_layers,
        channels=channels,
        num_heads=num_heads,
        head_features=head_features,
        multiplier=multiplier,
        use_context_time=True,
        context_features=None,
        context_embedding_features=context_embedding_features,
        embedding_max_length=embedding_max_length,
    )

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )
    '''
    diffusion.diffusion = KDiffusion(
        net=dit,  # <- use DiT here
        sigma_distribution=LogNormalDistribution(
            mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
        ),
        sigma_data=args.diffusion.dist.sigma_data,
        dynamic_threshold=0.0,
    )
    '''
    diffusion.diffusion = VKDiffusion(
            net=dit,
            sigma_distribution=LogNormalDistribution(
            mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
            ),
            min_snr_gamma=5.0,
            robust="huber",
            huber_delta=0.5,
        )
    # Keep a reference if other code expects .unet
    diffusion.unet = dit
    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd = MultiPeriodDiscriminator(),
            msd = MultiResSpecDiscriminator(),
        
            # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets



def load_checkpoint(model, optimizer, path, load_only_params=False, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    print('loading the ckpt using the correct function.')

    for key in model:
        if key in params and key not in ignore_modules:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                print(f'{key} key length: {len(model[key].state_dict().keys())}, state_dict key length: {len(state_dict.keys())}')
                for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
            print('%s loaded' % key)

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
        
    return model, optimizer, epoch, iters
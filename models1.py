#coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from typing import Optional
from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

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

class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


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

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)



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
        
       

def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    # t: [B, 1, 1] or [B, T, 1] in [0, 1]
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device) / half
    )
    args = t * freqs  # broadcast
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, T, dim] or [B, 1, dim]

class FlowMatchingDurationHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        t_emb_dim: Optional[int] = None,
        local_ctx: int = 32,            # local attention window
        max_dur: float = 50.0,          # hard cap (in frames)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.t_emb_dim = t_emb_dim or in_dim
        self.local_ctx = local_ctx
        self.max_dur = float(max_dur)

        self.x_proj = nn.Linear(1, in_dim)
        self.t_proj = nn.Linear(self.t_emb_dim, in_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * in_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(in_dim, 1)

    @staticmethod
    def _make_key_padding_mask(mask_bool: torch.Tensor) -> torch.Tensor:
        return mask_bool  # [B, T], True = PAD

    def _build_local_mask(
        self, T: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self.local_ctx is None or self.local_ctx <= 0:
            return None
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        allowed = dist <= self.local_ctx
        return ~allowed  # True = disallowed

    def forward(
        self,
        h: torch.Tensor,                 # [B, T, in_dim]
        x_t: torch.Tensor,               # [B, T, 1]
        t: torch.Tensor,                 # [B, 1, 1] or [B, T, 1]
        key_padding_mask: torch.Tensor,  # [B, T] True=PAD
    ) -> torch.Tensor:
        B, T, _ = h.shape
        x_emb = self.x_proj(x_t)
        t_emb = timestep_embedding(t.expand(B, T, 1), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        z = h + x_emb + t_emb

        attn_mask = self._build_local_mask(T, z.device)
        v = self.encoder(
            z,
            mask=attn_mask,
            src_key_padding_mask=self._make_key_padding_mask(key_padding_mask),
        )
        return self.out(v)  # [B, T, 1]

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,                 # [B, T, in_dim]
        key_padding_mask: torch.Tensor,  # [B, T]
        steps: int = 32,
        sigma: float = 0.7,
        clamp_max: Optional[float] = None,  # if None, uses self.max_dur
    ) -> torch.Tensor:
        B, T, _ = h.shape
        device = h.device
        x = torch.randn(B, T, 1, device=device) * sigma
        t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t = t_grid[i].view(1, 1, 1).expand(B, 1, 1)
            v = self.forward(h, x, t, key_padding_mask)
            x = x + v * dt

        dur = torch.exp(x).squeeze(-1)
        cap = float(self.max_dur if clamp_max is None else clamp_max)
        dur = torch.clamp(dur, min=1.0, max=cap)
        return dur

    def flow_matching_loss(
        self,
        h: torch.Tensor,                 # [B, T, in_dim]
        d_gt: torch.Tensor,              # [B, T] integer (or float) durations
        key_padding_mask: torch.Tensor,  # [B, T] True=PAD
        sigma: float = 1.0,
        aux_recon: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Enforce cap on targets during training
        d_gt_c = torch.clamp(d_gt.float(), min=1.0, max=self.max_dur)  # [B,T]
        z1 = torch.log(d_gt_c).unsqueeze(-1)                           # [B,T,1]
        z0 = torch.randn_like(z1) * sigma

        B, T, _ = z1.shape
        device = z1.device

        t = torch.rand(B, 1, 1, device=device)
        x_t = (1.0 - t) * z0 + t * z1
        v_star = z1 - z0

        v_pred = self.forward(h, x_t, t, key_padding_mask)  # [B, T, 1]

        not_pad = ~key_padding_mask
        mask = not_pad.unsqueeze(-1).float()  # [B, T, 1]

        # FM loss
        mse = ((v_pred - v_star) ** 2) * mask
        loss_fm = mse.sum() / mask.sum().clamp_min(1.0)

        # Aux reconstruction (also bounded so it never pushes beyond cap)
        loss_aux = torch.tensor(0.0, device=device)
        if aux_recon:
            z_hat = x_t + (1.0 - t) * v_pred
            z_hat = torch.clamp(
                z_hat,
                min=math.log(1.0),
                max=math.log(self.max_dur),
            )
            l1 = (torch.abs(z_hat - z1) * mask).sum() / mask.sum().clamp_min(1.0)
            loss_aux = l1

        return loss_fm, loss_aux
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
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
        
        
'''        
class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        d_hid: int,
        nlayers: int,
        max_dur: int = 50,  # unused for flow but kept for API compat
        dropout: float = 0.1,
        attn_heads: int = 8,
        attn_window: int = 5,
        use_rel_bias: bool = False,
    ):
        super().__init__()

        # IMPORTANT: DurationEncoder outputs [B, T_text, d_hid + style_dim]
        # because it concatenates style each block.
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )

        in_dim = d_hid + style_dim  # e.g. 512 + 128 = 640
        self.in_dim = in_dim

        # Flow head (kept intact)
        self.flow_head = FlowMatchingDurationHead(
            in_dim=in_dim,
            n_layers=4,
            n_heads=8,  # ensure in_dim % n_heads == 0 or change accordingly
            ff_mult=4,
            dropout=dropout,
        )

        # Shared trunk for F0/N (expects in_dim channels)
        self.shared = nn.LSTM(in_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)

        # F0 / N decoder stacks (unchanged)
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
        
        self.f0_mu = nn.Conv1d(d_hid // 2, 1, 1)
        self.f0_logsig = nn.Conv1d(d_hid // 2, 1, 1)
        self.f0_vuv = nn.Conv1d(d_hid // 2, 1, 1)

        self.N_mu = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_logsig = nn.Conv1d(d_hid // 2, 1, 1)

        # Cross-attention path to build frame-level prosody encodings
        self.en_channels = in_dim  # channels of token & frame encodings

        # Style-conditioned normalization layers (AdaLN) for q/k
        # If you have your own AdaLayerNorm replace SimpleAdaLayerNorm
        self.ca_q_norm = AdaLayerNorm(style_dim, self.en_channels)
        self.ca_k_norm = AdaLayerNorm(style_dim, self.en_channels)

        # Cross-attention module (q: frames, k/v: tokens)
        self.cross_attn = CrossAttention(d_model=self.en_channels, n_heads=attn_heads, dropout=dropout)

        self.attn_window = attn_window
        self.use_rel_bias = use_rel_bias

        # small conv post-processing for attended frame encodings
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

    def compute_en(
        self,
        d_tok: torch.Tensor,
        alignment: torch.Tensor,
        style: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build frame-level prosody encodings using cross-attention.
        d_tok: [B, T, C] token states (style-conditioned)
        alignment: [B, T, F] monotonic alignment
        style: [B, style_dim]
        text_mask: [B, T] True at padding
        Returns: en [B, C, F]
        """
        # Base (residual): aggregate tokens by alignment (matrix multiplication)
        # en_base: [B, C, F]
        en_base = torch.matmul(d_tok.transpose(1, 2), alignment)

        # Queries (frames) and keys (tokens): shapes for attention are [B, L, C]
        q0 = en_base.transpose(1, 2)  # [B, F, C]
        k0 = d_tok  # [B, T, C]

        # Style-conditioned normalization (applied per element over channels)
        q = self.ca_q_norm(q0, style)  # [B, F, C]
        k = self.ca_k_norm(k0, style)  # [B, T, C]

        # Build monotonic band mask: allowed tokens per frame
        attn_mask = build_monotonic_band_mask(alignment, text_mask, self.attn_window)
        # Optional relative (distance) bias
        rel_bias = None
        if self.use_rel_bias:
            with torch.no_grad():
                B, T, F = alignment.shape
                tau = alignment.argmax(dim=1)  # [B, F]
                t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
                dist = (t_idx - tau.unsqueeze(-1)).abs().float()  # [B, F, T]
                rel_bias = (-0.05 * dist).unsqueeze(1)  # [B, 1, F, T]

        # Cross-attend: q [B, F, C], k/v [B, T, C] -> out [B, F, C]
        en_attn = self.cross_attn(q, k, k, attn_mask=attn_mask, rel_bias=rel_bias)

        # post-process: [B, F, C] -> [B, C, F]
        en_attn = en_attn.transpose(1, 2)
        en_attn = self.en_post(en_attn)

        # residual combine and scale
        en = (en_base + en_attn) / math.sqrt(2.0)
        return en

    def forward(
        self,
        texts: torch.Tensor,  # d_en: [B, d_hid, T_text] or however DurationEncoder expects
        style: torch.Tensor,  # [B, style_dim]
        text_lengths: torch.Tensor,
        alignment: torch.Tensor,  # [B, T_text, T_frames/2]
        m: torch.Tensor,  # [B, T_text] True=PAD
        d_gt: torch.Tensor,  # [B, T_text] integer durations
    ):
        # Encode tokens with style conditioning
        # d_full: [B, T_text, d_hid + style_dim]
        d_full = self.text_encoder(texts, style, text_lengths, m)

        # Build enriched frame-level encoding using cross-attention
        en = self.compute_en(d_full, alignment, style, m)

        # Flow-matching loss on per-token durations (same as original)
        loss_fm, loss_aux = self.flow_head.flow_matching_loss(
            d_full, d_gt, key_padding_mask=m
        )

        return en, loss_fm, loss_aux
    
    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor):
        # x: [B, in_dim, T_frames/2]  (same as before)
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
   
        
    def F0Ntrain(self, en: torch.Tensor, s: torch.Tensor):
        """
        Predicts F0 and Norm distributions from frame-level encodings.

        Args:
            en (torch.Tensor): Frame-level encodings [B, C, F].
            s (torch.Tensor): Style vector [B, style_dim].

        Returns:
            Tuple: A tuple containing:
                - F0_pred (torch.Tensor): The predicted F0 mean for synthesis [B, F].
                - N_pred (torch.Tensor): The predicted Norm mean for synthesis [B, F].
                - extras (dict): A dictionary with all distribution parameters for loss calculation.
        """
        # The shared LSTM expects input of shape [B, Seq_Len, Features]
        # en is [B, C, F], so we transpose it to [B, F, C]
        x, _ = self.shared(en.transpose(1, 2))

        # Transpose back to [B, C, F] for the 1D convolutional blocks
        x = x.transpose(1, 2)

        # --- F0 Branch ---
        F0_hidden = x
        for block in self.F0:
            F0_hidden = block(F0_hidden, s)

        # Predict F0 distribution parameters from the hidden state
        f0_mu = self.f0_mu(F0_hidden)
        f0_logsig = self.f0_logsig(F0_hidden)
        vuv_logits = self.f0_vuv(F0_hidden)

        # --- Norm (N) Branch ---
        N_hidden = x
        for block in self.N:
            N_hidden = block(N_hidden, s)

        # Predict Norm distribution parameters from the hidden state
        N_mu = self.N_mu(N_hidden)
        N_logsig = self.N_logsig(N_hidden)

        # For synthesis, we use the mean (mu) of the predicted distributions.
        # We squeeze the channel dimension (dim=1) to get [B, F].
        F0_pred = f0_mu.squeeze(1)
        N_pred = N_mu.squeeze(1)

        # Pack the other parameters into a dictionary for easy access in the loss function.
        extras = {
            "f0_mu": f0_mu,
            "f0_logsig": f0_logsig,
            "vuv_logits": vuv_logits,
            "N_mu": N_mu,
            "N_logsig": N_logsig,
        }

        return F0_pred, N_pred, extras
   
    @torch.no_grad()
    def sample_durations(
        self,
        d_full: torch.Tensor,
        texts: torch.Tensor,
        style: torch.Tensor,
        text_lengths: torch.Tensor,
        m: torch.Tensor,
        steps: int = 32,
        sigma: float = 1.0,
        clamp_max: float = 300.0,
    ) -> torch.Tensor:
        # d_full: [B, T_text, in_dim]
        d_hat = self.flow_head.sample(
            d_full, m, steps=steps, sigma=sigma, clamp_max=clamp_max
        )
        # round to integers >= 1
        d_hat = torch.round(d_hat).clamp_(min=1.0)
        return d_hat  # [B, T_text]        
     
class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        d_hid: int,
        nlayers: int,
        max_dur: int = 50,  # unused now, kept for API compatibility
        dropout: float = 0.1,
    ):
        super().__init__()

        # IMPORTANT: DurationEncoder outputs [B, T, d_hid + style_dim]
        # because it concatenates style each block.
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

        in_dim = d_hid + style_dim  # 512 + 128 = 640
        self.flow_head = FlowMatchingDurationHead(
            in_dim=in_dim,
            n_layers=4,
            n_heads=8,  # 640 / 8 = 80 per head
            ff_mult=4,
            dropout=dropout,
        )

        # Shared trunk for F0/N (unchanged, expects in_dim channels)
        self.shared = nn.LSTM(
            in_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )

        self.F0 = nn.ModuleList()
        self.F0.append(
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
        )
        self.F0.append(
            AdainResBlk1d(
                d_hid,
                d_hid // 2,
                style_dim,
                upsample=True,
                dropout_p=dropout,
            )
        )
        self.F0.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.N = nn.ModuleList()
        self.N.append(
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
        )
        self.N.append(
            AdainResBlk1d(
                d_hid,
                d_hid // 2,
                style_dim,
                upsample=True,
                dropout_p=dropout,
            )
        )
        self.N.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(
        self,
        texts: torch.Tensor,  # d_en: [B, d_hid, T_text]
        style: torch.Tensor,  # [B, style_dim=128]
        text_lengths: torch.Tensor,
        alignment: torch.Tensor,  # [B, T_text, T_frames/2]
        m: torch.Tensor,  # [B, T_text] True=PAD
        d_gt: torch.Tensor,  # [B, T_text] integer durations
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode tokens with style conditioning
        # d_full: [B, T_text, d_hid + style_dim]
        d_full = self.text_encoder(texts, style, text_lengths, m)

        # Frame-level prosody features for F0/N heads:
        # [B, d_hid+style, T_text]^T @ [B, T_text, T_frames/2]
        en = d_full.transpose(-1, -2) @ alignment  # [B, in_dim, T_frames/2]

        # Flow-Matching loss on per-token durations
        loss_fm, loss_aux = self.flow_head.flow_matching_loss(
            d_full, d_gt, key_padding_mask=m
        )

        return en, loss_fm, loss_aux

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor):
        # x: [B, in_dim, T_frames/2]  (same as before)
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

    @torch.no_grad()
    def sample_durations(
        self,
        texts: torch.Tensor,  # d_en: [B, d_hid, T_text]
        style: torch.Tensor,  # [B, style_dim]
        text_lengths: torch.Tensor,
        m: torch.Tensor,  # [B, T_text] True=PAD
        steps: int = 32,
        sigma: float = 1.0,
        clamp_max: float = 300.0,
    ) -> torch.Tensor:
        # d_full: [B, T_text, in_dim]
        d_full = self.text_encoder(texts, style, text_lengths, m)
        d_hat = self.flow_head.sample(
            d_full, m, steps=steps, sigma=sigma, clamp_max=clamp_max
        )
        # round to integers >= 1
        d_hat = torch.round(d_hat).clamp_(min=1.0)
        return d_hat  # [B, T_text]

    
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

        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

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
        self.F0 = nn.ModuleList()
        self.F0.append(
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
        )
        self.F0.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.F0.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.N = nn.ModuleList()
        self.N.append(
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
        )
        self.N.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.N.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # New: cross attention to build frame-level prosody encodings
        self.en_channels = d_hid + style_dim  # 640 in your setup

        # Cross-attention path should use that channel size
        self.ca_q_norm = AdaLayerNorm(style_dim, self.en_channels)
        self.ca_k_norm = AdaLayerNorm(style_dim, self.en_channels)

        self.cross_attn = CrossAttention(
            d_model=self.en_channels,  # 640
            n_heads=attn_heads,        # make sure 640 % attn_heads == 0 (e.g., 4, 5, 8, 10)
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
            weight_norm(
                nn.Conv1d(self.en_channels, self.en_channels, kernel_size=1)
            ),
        )
    def compute_en(self, d_tok, alignment, style, text_mask):
        """
        d_tok: [B, T, C] token states (style-conditioned)
        alignment: [B, T, F] monotonic alignment
        style: [B, S]
        text_mask: [B, T] True at padding
        Returns: en [B, C, F]
        """
        # Base aggregation as residual
        en_base = torch.matmul(d_tok.transpose(1, 2), alignment)  # [B, C, F]

        # Build queries from the same base (frame-level)
        q0 = en_base.transpose(1, 2)  # [B, F, C]
        k0 = d_tok  # [B, T, C]

        # Style-conditioned layer norms (AdaLayerNorm expects [B, C, L])
        q = self.ca_q_norm(q0, style)
        k = self.ca_k_norm(k0, style)

        # Monotonic band mask
        attn_mask = build_monotonic_band_mask(
            alignment, text_mask, self.attn_window
        )  # [B, 1, F, T]

        # Optional: simple ALiBi-like negative bias to prefer nearby tokens
        rel_bias = None
        if self.use_rel_bias:
            with torch.no_grad():
                B, T, F = alignment.shape
                tau = alignment.argmax(dim=1)  # [B, F]
                t_idx = torch.arange(
                    T, device=alignment.device
                ).view(1, 1, T)
                dist = (t_idx - tau.unsqueeze(-1)).abs().float()
                # slope tuned small to avoid over-biasing
                rel_bias = (-0.05 * dist).unsqueeze(1)  # [B, 1, F, T]

        en_attn = self.cross_attn(q, k, k, attn_mask=attn_mask, rel_bias=rel_bias)
        # [B, F, C] -> [B, C, F] and smooth
        en_attn = en_attn.transpose(1, 2)
        en_attn = self.en_post(en_attn)

        # Residual combine
        en = (en_base + en_attn) / math.sqrt(2.0)  
        return en

    def forward(self, texts, style, text_lengths, alignment, m):
       
        d_tok = self.text_encoder(texts, style, text_lengths, m)  # [B, T, C]

        # Duration head (same as your original code)
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d_tok, input_lengths, batch_first=True, enforce_sorted=False
        )
        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad

        duration = self.duration_proj(
            nn.functional.dropout(x, 0.5, training=self.training)
        )



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
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
        

'''
class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        
        self.text_encoder = DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(-1, -2) @ alignment)

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
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask




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


class MHSA(nn.Module):
    def __init__(self, hidden: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T, C]
        key_padding_mask: torch.Tensor,     # [B, T] bool, True = pad
        local_attn_mask: torch.Tensor,      # [T, T] bool, True = ZABRONIONE (poza oknem)
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1) Złóż maski do per-batch [B, T, T] (True = maskować)
        if local_attn_mask is not None:
            full = local_attn_mask.to(torch.bool).unsqueeze(0).expand(B, T, T).clone()
        else:
            full = torch.zeros(B, T, T, dtype=torch.bool, device=device)

        if key_padding_mask is not None:
            # maskujemy KLUCZE na padach
            full |= key_padding_mask.unsqueeze(1).expand(B, T, T)

        # 2) Zagwarantuj co najmniej 1 dozwolony element w wierszu (diagonała)
        idx = torch.arange(T, device=device)
        full[:, idx, idx] = False

        # 3) Zamień na addytywną maskę float (0 = dozwolone, −big = karane)
        attn_bias = torch.zeros(B, T, T, dtype=dtype, device=device)
        neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
        attn_bias.masked_fill_(full, neg)

        # 4) Rozszerz na głowy: [B*n_heads, T, T]
        attn_bias = (
            attn_bias.unsqueeze(1)
            .repeat(1, self.n_heads, 1, 1)
            .reshape(B * self.n_heads, T, T)
        )

        # 5) MHA z maską addytywną; key_padding_mask już wliczona
        y, _ = self.mha(
            x, x, x,
            attn_mask=attn_bias,
            key_padding_mask=None,
            need_weights=False,
        )

        # 6) Wyzeruj wyjście na pozycjach pad (bezpiecznik)
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return self.dropout(y)


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
        self.mha = MHSA(hidden, n_heads=n_heads, dropout=dropout)

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

''''
class DurationEncoder(nn.Module):
    def __init__(
        self,
        sty_dim: int,
        d_model: int,
        nlayers: int,
        n_heads: int = 8,
        conv_kernel: int = 31,
        local_attn_ctx: int = 128,
        dropout: float = 0.1,
        in_channels: int | None = None,
        concat_style_out: bool = True,  # NEW: keep old API shapes
    ):
        super().__init__()
        self.d_model = d_model
        self.local_attn_ctx = local_attn_ctx
        self.concat_style_out = concat_style_out

        self.in_proj = (
            nn.Linear(in_channels, d_model)
            if in_channels and in_channels != d_model
            else None
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    style_dim=sty_dim,
                    hidden=d_model,
                    n_heads=n_heads,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(nlayers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def _build_local_attn_mask(self, T: int, device) -> torch.Tensor:
        # True = disallowed attention
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        local = dist <= self.local_attn_ctx
        return ~local  # [T, T] bool
    def forward(
        self,
        x: torch.Tensor,       # [B, C, T]
        style: torch.Tensor,   # [B, sty_dim]
        text_lengths: torch.Tensor,
        m: torch.Tensor,       # [B, T] True = pad
    ) -> torch.Tensor:
        B, C, T = x.shape
        x = x.transpose(1, 2)  # [B, T, C]
        if self.in_proj is not None:
            x = self.in_proj(x)
        
        attn_mask = self._build_local_attn_mask(T, x.device)  # [T, T] bool
        key_padding_mask = m  # [B, T] bool
        for blk in self.blocks:
            x = blk(x, style, key_padding_mask, attn_mask)
            

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # NEW: concat style to match old shape [B, T, d_model + sty_dim]
        if self.concat_style_out:
            s = style.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, sty_dim]
            x = torch.cat([x, s], dim=-1)

        return self.dropout(x)

    def inference(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, T, C]
        if self.in_proj is not None:
            x = self.in_proj(x)
        T = x.size(1)
        attn_mask = self._build_local_attn_mask(T, x.device)
        key_padding_mask = torch.zeros(
            x.size(0), T, dtype=torch.bool, device=x.device
        )
        for blk in self.blocks:
            x = blk(x, style, key_padding_mask, attn_mask)

        if self.concat_style_out:
            s = style.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, s], dim=-1)

        return x
'''
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
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
    else:
        from Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
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
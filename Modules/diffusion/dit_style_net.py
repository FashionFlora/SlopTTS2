# file: Modules/diffusion/dit_style_net.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

# Utilities: rand_bool and FixedEmbedding should match your existing helpers
# If you already have them imported elsewhere, remove duplicates.
def rand_bool(shape, proba, device=None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E] – only T and B used for positional indices
        b, t = x.shape[0], x.shape[1]
        assert t <= self.max_length, "Sequence length exceeds max_length"
        pos = torch.arange(t, device=x.device)
        pe = self.embedding(pos)  # [T, E]
        return repeat(pe, "t e -> b t e", b=b)


# --------- DiT core blocks (AdaLN-Zero) ---------
class AdaLNZero(nn.Module):
    """
    AdaLayerNorm-Zero: LayerNorm followed by (1 + gamma) * x + beta,
    with gamma, beta produced from conditioning vector (here, time embedding).
    Initialized to zeros so residual paths start as identity.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Linear(cond_dim, dim, bias=True)
        self.beta = nn.Linear(cond_dim, dim, bias=True)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], c: [B, Dc]
        x = self.norm(x)
        g = self.gamma(c)[:, None, :]  # [B, 1, D]
        b = self.beta(c)[:, None, :]
        return x * (1 + g) + b


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        cond_dim: int = 256,
        attn_stochastic_depth: float = 0.0,
        mlp_stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self.ada1 = AdaLNZero(dim, cond_dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, dropout=dropout)
        self.ada2 = AdaLNZero(dim, cond_dim)
        self.mlp = MLP(dim, mult=mlp_mult, dropout=dropout)

        # Stochastic depth (DropPath) probabilities
        self.p_attn = attn_stochastic_depth
        self.p_mlp = mlp_stochastic_depth

    def drop_path(self, x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if p == 0.0 or not training:
            return x
        keep = 1 - p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], c: [B, Dc]
        h = self.ada1(x, c)
        h = self.attn(h)
        x = x + self.drop_path(h, self.p_attn, self.training)
        h = self.ada2(x, c)
        h = self.mlp(h)
        x = x + self.drop_path(h, self.p_mlp, self.training)
        return x


# --------- Time embedding from c_noise (EDM) ---------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        # c_noise: [B] (log-sigma * 0.25 in KDiffusion)
        return self.net(c_noise[:, None])  # [B, dim]


# --------- DiT-style net for style vector diffusion ---------
class DiTStyleNet(nn.Module):
    """
    Drop-in replacement for your Transformer/StyleTransformer backbone.
    Compatible with KDiffusion:
      x_in: [B, C, 1] or [B, C] -> we treat as [B, C]
      c_noise: [B] (Karras EDM noise conditioning, log-sigma * 0.25)
      embedding: [B, T, E] (BERT sequence)
    Returns predicted x_pred in EDM formulation (same shape as x_in).
    """

    def __init__(
        self,
        channels: int = 256,  # style dim (s_ref|s)
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        txt_emb_dim: int = 512,  # BERT hidden size
        max_txt_len: int = 512,
        stochastic_depth: float = 0.0,  # linearly scaled across layers
        cfg_dropout_prob: float = 0.1,  # classifier-free masking prob
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_pos = use_positional_encoding

        # Project style vector to token
        self.x_proj = nn.Linear(channels, d_model)
        # Learnable [STYLE] token bias (helps stability)
        self.style_bias = nn.Parameter(torch.zeros(d_model))

        # Project text embeddings to model dim
        self.txt_proj = nn.Linear(txt_emb_dim, d_model)

        # Fixed positional embeddings for text tokens
        self.pos_emb = FixedEmbedding(max_length=max_txt_len, features=d_model)

        # Time embedding from c_noise
        self.time_mlp = TimeEmbedding(dim=d_model)

        # DiT blocks with AdaLN-Zero and stochastic depth
        blocks = []
        for i in range(num_layers):
            p = stochastic_depth * float(i) / max(1, num_layers - 1)
            blocks.append(
                DiTBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                    cond_dim=d_model,
                    attn_stochastic_depth=p,
                    mlp_stochastic_depth=p,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Output head maps style token back to channels
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, channels),
        )

    def forward(
        self,
        x_in: torch.Tensor,               # [B, C] or [B, C, 1]
        c_noise: torch.Tensor,            # [B] (EDM conditioning)
        *,
        embedding: torch.Tensor,          # [B, T, E]
        embedding_mask_proba: float = 0.0,
        embedding_scale: float = 1.0,
    ) -> torch.Tensor:
        # Normalize x_in shape
        if x_in.dim() == 3:
            if x_in.shape[1] == 1 and x_in.shape[2] == self.channels:
                # [B, 1, C] -> squeeze dim 1
                x_in = x_in.squeeze(1)  # [B, C]
            elif x_in.shape[2] == 1 and x_in.shape[1] == self.channels:
                # [B, C, 1] -> squeeze last dim
                x_in = x_in.squeeze(-1)  # [B, C]
            else:
                raise AssertionError(
                    f"Unexpected x_in shape {tuple(x_in.shape)} for channels={self.channels}"
                )
        assert x_in.dim() == 2 and x_in.shape[1] == self.channels

        b, device = x_in.shape[0], x_in.device

        # Classifier-free guidance: mask text with prob
        txt = embedding
        if embedding_mask_proba > 0.0 or self.cfg_dropout_prob > 0.0:
            proba = max(embedding_mask_proba, self.cfg_dropout_prob)
            mask = rand_bool((b, 1, 1), proba, device=device)
            # Use zero conditioning (or learned null) by zeroing txt
            txt = torch.where(mask, torch.zeros_like(txt), txt)

        # Optionally compute masked pass for CFG
        if embedding_scale != 1.0:
            out = self._run(x_in, c_noise, txt)
            out_null = self._run(x_in, c_noise, torch.zeros_like(txt))
            return out_null + (out - out_null) * embedding_scale
        else:
            return self._run(x_in, c_noise, txt)

    def _run(
        self, x_in: torch.Tensor, c_noise: torch.Tensor, txt: torch.Tensor
    ) -> torch.Tensor:
        b, device = x_in.shape[0], x_in.device

        # Build token sequence: [STYLE] + text tokens
        style_tok = self.x_proj(x_in) + self.style_bias  # [B, D]
        style_tok = style_tok[:, None, :]  # [B, 1, D]

        txt_tok = self.txt_proj(txt)  # [B, T, D]
        if self.use_pos:
            txt_tok = txt_tok + self.pos_emb(txt_tok)  # [B, T, D]

        tokens = torch.cat([style_tok, txt_tok], dim=1)  # [B, 1+T, D]

        # Time conditioning
        t = self.time_mlp(c_noise)  # [B, D]

        # DiT blocks
        h = tokens
        for blk in self.blocks:
            h = blk(h, t)

        # Extract final style token and map to channels
        style_out = h[:, 0, :]  # [B, D]
        x_pred = self.head(style_out)  # [B, C]
        return x_pred
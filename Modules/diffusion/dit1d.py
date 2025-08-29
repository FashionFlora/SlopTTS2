# Modules/diffusion/dit1d.py
# Refactored 1D DiT with Multi-Query Attention and Adaptive Layer Normalization
# Drop-in replacements for Transformer1d and StyleTransformer1d
# AdaLNZero (two separate linears for gamma and beta) used for adaptive LN.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# --- Helpers (from your provided code) ---

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def rand_bool(shape, proba, device=None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(self.half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        freqs = x * self.weights.view(1, -1) * 2 * torch.pi
        fouriered = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        fouriered = torch.cat([x, fouriered], dim=-1)
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[0], x.shape[1]
        assert L <= self.max_length, "Input sequence length must be <= max_length"
        pos = torch.arange(L, device=x.device)
        pe = self.embedding(pos)
        pe = pe.unsqueeze(0).expand(B, L, -1)
        return pe

# --- Adaptive LayerNorm: AdaLNZero (two linears for gamma and beta) ---

class AdaLNZero(nn.Module):

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

# --- (Keep original single-linear AdaLayerNorm as an option) ---

class AdaLayerNorm(nn.Module):
    """
    Alternative fused single-linear implementation (kept for compatibility).
    """
    def __init__(self, cond_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], cond: [B, cond_dim]
        cond = cond.unsqueeze(1)  # [B, 1, cond_dim] for broadcasting
        gamma_beta = self.fc(cond)  # [B, 1, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, 1, C], [B, 1, C]
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta

# --- Attention, MLP, Blocks ---

class MQAttention(nn.Module):
    def __init__(
        self,
        features: int,
        head_features: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = head_features
        self.norm = nn.LayerNorm(features)
        self.to_q = nn.Linear(features, num_heads * head_features, bias=False)
        # Multi-query: shared k/v per head set (no separate heads for kv)
        self.to_kv = nn.Linear(features, 2 * head_features, bias=False)
        self.scale = head_features ** -0.5
        self.to_out = nn.Linear(num_heads * head_features, features)

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        k, v = self.to_kv(x_norm).chunk(2, dim=-1)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)

        attn = torch.einsum("b h t d, b s d -> b h t s", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b h t s, b s d -> b h t d", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out) + x

class MLP(nn.Module):
    def __init__(self, features: int, multiplier: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, features * multiplier),
            nn.GELU(),
            nn.Linear(features * multiplier, features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DiTBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        cond_dim: int,
        use_adalnzero: bool = True,
    ):
        super().__init__()
        # choose AdaLNZero by default, fallback to AdaLayerNorm if disabled
        ln_class = AdaLNZero if use_adalnzero else AdaLayerNorm

        # adapter expects (dim, cond_dim) signature for AdaLNZero
        if ln_class is AdaLNZero:
            self.attn_mod = ln_class(dim=features, cond_dim=cond_dim)
            self.ffn_mod = ln_class(dim=features, cond_dim=cond_dim)
        else:
            self.attn_mod = ln_class(cond_dim=cond_dim, channels=features)
            self.ffn_mod = ln_class(cond_dim=cond_dim, channels=features)

        self.attn = MQAttention(
            features=features, head_features=head_features, num_heads=num_heads
        )
        self.ffn = MLP(features=features, multiplier=multiplier)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.attn_mod(x, cond))
        x = self.ffn(self.ffn_mod(x, cond)) + x
        return x

# --- Main Model Class ---

class _BaseDiT1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,  # Dimension of the style vector
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: int,  # Dimension of BERT embedding
        embedding_max_length: int = 512,
        use_style_conditioning: bool = False,
        style_features: Optional[int] = None,  # Dimension of reference style features
        use_adalnzero: bool = True,  # toggle to use AdaLNZero or fused AdaLayerNorm
    ):
        super().__init__()
        self.use_style_conditioning = use_style_conditioning
        self.use_adalnzero = use_adalnzero

        token_dim = channels + context_embedding_features
        cond_dim = token_dim

        self.to_time = nn.Sequential(
            TimePositionalEmbedding(dim=channels, out_features=cond_dim),
            nn.GELU(),
        )

        if use_style_conditioning:
            assert exists(style_features), "style_features dimension must be provided"
            self.to_style = nn.Sequential(
                nn.Linear(in_features=style_features, out_features=cond_dim),
                nn.GELU(),
            )

        self.mapping = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(
                features=token_dim,
                num_heads=num_heads,
                head_features=head_features,
                multiplier=multiplier,
                cond_dim=cond_dim,
                use_adalnzero=use_adalnzero,
            ) for _ in range(num_layers)
        ])

        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features
        )

        self.out_norm = nn.LayerNorm(token_dim)
        self.to_out = nn.Linear(in_features=token_dim, out_features=channels)

    def get_conditioning(self, time, features):
        cond = self.to_time(time)
        if self.use_style_conditioning:
            assert exists(features), "features must be provided for style conditioning"
            cond = cond + self.to_style(features)
        return self.mapping(cond)

    def run(self, x, time, embedding, features):
        # x: [B, 1, C], embedding: [B, L, E]
        L = embedding.shape[1]
        x_tokens = x.expand(-1, L, -1)
        tokens = torch.cat([x_tokens, embedding], dim=-1)

        cond = self.get_conditioning(time, features)

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.out_norm(tokens)
        tokens = tokens.mean(axis=1)  # Average pool over sequence length
        out = self.to_out(tokens)
        return out.unsqueeze(1)  # [B, 1, C]

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        embedding_mask_proba: float = 0.0,
        embedding: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        embedding_scale: float = 1.0,
    ) -> torch.Tensor:
        b, device = x.shape[0], x.device
        assert exists(embedding), "BERT embedding must be provided"

        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            batch_mask = rand_bool((b, 1, 1), proba=embedding_mask_proba, device=device)
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)

# --- Drop-in Replacement Classes ---

class Transformer1d(_BaseDiT1d):
    """ API-compatible DiT1d for unconditional/time-conditional generation """
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
        use_adalnzero: bool = True,
        **kwargs  # Ignore unused legacy arguments
    ):
        assert exists(context_embedding_features)
        super().__init__(
            num_layers=num_layers,
            channels=channels,
            num_heads=num_heads,
            head_features=head_features,
            multiplier=multiplier,
            context_embedding_features=context_embedding_features,
            embedding_max_length=embedding_max_length,
            use_style_conditioning=False,
            use_adalnzero=use_adalnzero,
        )

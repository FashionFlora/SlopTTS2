# Modules/diffusion/dit1d.py
# Single-version DiT1d with:
# - cross-attention (context k/v + norm_context)
# - full multi-head K/V (no multi-query)
# - optional bucketed Relative Position Bias
# - AdaLNZero adaptive LayerNorm (gamma & beta)
# - simplified, single codepath (old unused code removed)

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, einsum

# --- Helpers --------------------------------------------------------------

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

# --- Time / Fixed embeddings ----------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0, "dim must be even"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B] or [B,] continuous time
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
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

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, E] -- we use L
        batch_size, length, device = x.shape[0], x.shape[1], x.device
        assert length <= self.max_length, "Input sequence length must be <= max_length"
        pos = torch.arange(length, device=device)
        pe = self.embedding(pos)  # [L, E]
        pe = repeat(pe, "n d -> b n d", b=batch_size)
        return pe

# --- Adaptive LayerNorm (AdaLNZero) --------------------------------------

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm where scale (gamma) and shift (beta) are produced
    by two separate linears from conditioning vector.
    x: [B, T, D], c: [B, Dc]
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

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.norm(x)
        g = self.gamma(c).unsqueeze(1)  # [B, 1, D]
        b = self.beta(c).unsqueeze(1)
        return x * (1 + g) + b

# --- Relative Position Bias -----------------------------------------------

class RelativePositionBias(nn.Module):
    """
    Bucketed relative position bias (T5-like). Returns [1, heads, q, k].
    """
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int):
        # Implementation follows bucketization pattern used in many transformer impls
        num_buckets = int(num_buckets)
        max_distance = int(max_distance)
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)
        max_exact = max(1, num_buckets // 2)
        is_small = n < max_exact

        # compute val for large distances (use safe clamped float for log)
        n_float = n.float().clamp(min=1.0)
        denom = math.log(max_distance / max_exact) if max_distance > max_exact else 1.0
        # avoid div by zero; if denom is 0 fallback to linear mapping
        if denom == 0:
            val_if_large = n
        else:
            val_if_large = (
                max_exact
                + (
                    (torch.log(n_float / max_exact) / denom)
                    * (num_buckets - max_exact)
                )
            ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) -> Tensor:
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(num_queries, dtype=torch.long, device=device)
        k_pos = torch.arange(num_keys, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")  # [q, k]
        rp_bucket = self._relative_position_bucket(rel_pos, self.num_buckets * 2, self.max_distance)
        # rp_bucket shape [q, k], embed -> [q, k, heads]
        bias = self.relative_attention_bias(rp_bucket)
        bias = rearrange(bias, "q k h -> 1 h q k")
        return bias

# --- Attention components -------------------------------------------------

def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid),
        nn.GELU(),
        nn.Linear(in_features=mid, out_features=features),
    )

class AttentionBase(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_features: int,
        out_features: Optional[int] = None,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = head_features
        self.scale = head_features ** -0.5
        mid = num_heads * head_features
        self.use_rel_pos = use_rel_pos

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance), (
                "rel_pos_num_buckets and rel_pos_max_distance must be provided when use_rel_pos=True"
            )
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )

        if out_features is None:
            out_features = mid
        self.to_out = nn.Linear(in_features=mid, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q,k,v: [B, Nq, mid], [B, Nk, mid], [B, Nk, mid]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        sim = einsum("b h n d, b h m d -> b h n m", q, k)  # [B, H, Nq, Nk]
        if self.use_rel_pos:
            bias = self.rel_pos(sim.shape[-2], sim.shape[-1])  # [1, H, Nq, Nk]
            sim = sim + bias
        sim = sim * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h n m, b h m d -> b h n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = head_features
        mid = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.Identity()
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(in_features=features, out_features=mid, bias=False)
        # full multi-head K/V (per-head)
        self.to_kv = nn.Linear(in_features=context_features, out_features=mid * 2, bias=False)

        self.base = AttentionBase(
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        context = default(context, x)
        
        # Oczekujemy, że `x` zostało już znormalizowane przez AdaLNZero w
        # DiTBlock. Nie re-normalizujemy queries, żeby nie tracić efektu
        # modulacji (gamma/beta).
        q_in = x  # no extra LayerNorm on queries
        
        # Dla kontekstu: normalizujemy tylko wtedy, gdy jest to inny tensor niż
        # queries (czyli cross-attention). Jeśli context domyślnie == x (self-attn),
        # to nie chcemy dodatkowej normalizacji.
        if context is x:
            context_norm = context
        else:
            context_norm = self.norm_context(context)
        
        q = self.to_q(q_in)
        k, v = torch.chunk(self.to_kv(context_norm), chunks=2, dim=-1)
        out = self.base(q, k, v)
        return out
# --- Transformer / Block -------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_cross_attention = exists(context_features) and context_features > 0

        # self-attention (context_features defaults to features => self-attn)
        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention and exists(context):
            x = self.cross_attention(x, context=context) + x
        x = self.feed_forward(x) + x
        return x

class DiTBlock(nn.Module):
    """
    DiT-style block with AdaLNZero applied before attention/ffn.
    """
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        cond_dim: int,
        context_features: Optional[int] = None,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.attn_mod = AdaLNZero(dim=features, cond_dim=cond_dim)
        self.ffn_mod = AdaLNZero(dim=features, cond_dim=cond_dim)
        self.block = TransformerBlock(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            multiplier=multiplier,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
            context_features=context_features,
        )

    def forward(self, x: Tensor, cond: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # Pre-norm with conditioning, then block
        x = self.block(self.attn_mod(x, cond), context=context) + x
        x = self.ffn_mod(x, cond)
        x = self.block.feed_forward(x) + x
        return x

# --- Main Model -----------------------------------------------------------

class _BaseDiT1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,  # token dimension (channels of x)
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: int,  # dimension of text/visual embedding used as context
        embedding_max_length: int = 512,
        use_style_conditioning: bool = False,
        style_features: Optional[int] = None,
        use_adalnzero: bool = True,  # kept for compatibility, AdaLNZero is used
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        assert exists(context_embedding_features), "context_embedding_features must be provided"
        self.use_style_conditioning = use_style_conditioning

        # token features are just channels now (we use cross-attention for embeddings)
        token_dim = channels
        cond_dim = token_dim

        # time + style mapping -> cond vector of size cond_dim
        self.to_time = nn.Sequential(
            TimePositionalEmbedding(dim=channels, out_features=cond_dim),
            nn.GELU(),
        )

        if use_style_conditioning:
            assert exists(style_features), "style_features must be provided when use_style_conditioning is True"
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

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance), (
                "rel_pos_num_buckets and rel_pos_max_distance required when use_rel_pos=True"
            )

        # build blocks: each block will have cross-attention to the context embeddings
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    features=token_dim,
                    num_heads=num_heads,
                    head_features=head_features,
                    multiplier=multiplier,
                    cond_dim=cond_dim,
                    context_features=context_embedding_features,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for _ in range(num_layers)
            ]
        )

        self.fixed_embedding = FixedEmbedding(max_length=embedding_max_length, features=context_embedding_features)

        self.out_norm = nn.LayerNorm(token_dim)
        self.to_out = nn.Linear(in_features=token_dim, out_features=channels)

    def get_conditioning(self, time: Tensor, features: Optional[Tensor]):
        cond = self.to_time(time)
        if self.use_style_conditioning:
            assert exists(features), "style features must be provided"
            cond = cond + self.to_style(features)
        return self.mapping(cond)

    def run(self, x: Tensor, time: Tensor, embedding: Tensor, features: Optional[Tensor]):
        # x: [B, 1, C], embedding: [B, L, E]
        B, L = embedding.shape[0], embedding.shape[1]
        x_tokens = x.expand(-1, L, -1)  # [B, L, C]
        tokens = x_tokens  # we use cross-attention to conditioning embeddings

        cond = self.get_conditioning(time, features)  # [B, cond_dim]

        for block in self.blocks:
            tokens = block(tokens, cond=cond, context=embedding)

        tokens = self.out_norm(tokens)
        tokens = tokens.mean(dim=1)  # [B, C]
        out = self.to_out(tokens)  # [B, C]
        return out.unsqueeze(1)  # [B, 1, C]

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        embedding_mask_proba: float = 0.0,
        embedding: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
        embedding_scale: float = 1.0,
    ) -> Tensor:
        b, device = x.shape[0], x.device
        assert exists(embedding), "context embedding must be provided"

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

# --- Public wrapper -------------------------------------------------------

class Transformer1d(_BaseDiT1d):
    """API-compatible DiT1d (single unified version with cross-attention & relpos)."""
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
        use_adalnzero: bool = True,  # kept for API compat, AdaLNZero is used
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        **kwargs,
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
            style_features=None,
            use_adalnzero=use_adalnzero,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )


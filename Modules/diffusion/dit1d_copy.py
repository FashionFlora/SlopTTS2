# Modules/diffusion/dit1d.py
# DiT1d z cross-attention, full multi-head K/V, AdaLNZero i opcjonalnym RoPE.
from typing import Optional

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor, einsum


# --- Helpers ----------------------------------------------------------------

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


# --- Time / Fixed embeddings ------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """Continuous learned positional embedding for time (Fourier features)."""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0, "dim must be even"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B] (per-batch scalar time)
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
    """Fixed learned positional embeddings (used as fallback/masked embedding)."""

    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, E] (we only need L)
        batch_size, length, device = x.shape[0], x.shape[1], x.device
        assert length <= self.max_length, "Input sequence length must be <= max_length"
        pos = torch.arange(length, device=device)
        pe = self.embedding(pos)  # [L, E]
        pe = repeat(pe, "n d -> b n d", b=batch_size)
        return pe


# --- AdaLNZero ---------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with two separate linears producing gamma and beta.
    x: [B, T, D], cond: [B, Dc]
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


# --- Rotary embeddings (RoPE) ------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings helper. Stores inverse freqs buffer.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "rotary dim must be even"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_sin_cos(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        sin = torch.sin(sinusoid_inp).to(dtype=dtype, device=device)
        cos = torch.cos(sinusoid_inp).to(dtype=dtype, device=device)
        return sin, cos


def apply_rotary_pos_emb(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    Apply rotary embedding to x.
    x: [B, H, N, D], sin/cos: [N, D//2]
    returns x with same shape
    """
    b, h, n, d = x.shape
    assert d % 2 == 0
    # sin, cos expected shape [N, D//2], expand to [1,1,N,D//2]
    sin = sin[None, None, :, :].to(dtype=x.dtype, device=x.device)
    cos = cos[None, None, :, :].to(dtype=x.dtype, device=x.device)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).reshape(b, h, n, d)
    return x_rot


# --- Feed-forward & Attention base -----------------------------------------

def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid),
        nn.GELU(),
        nn.Linear(in_features=mid, out_features=features),
    )


class AttentionBase(nn.Module):
    """
    Core attention (multi-head), optionally applies RoPE to q/k.
    q/k/v expected pre-projected to shape [B, N, num_heads*head_features]
    """

    def __init__(
        self,
        num_heads: int,
        head_features: int,
        out_features: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = head_features
        self.scale = head_features ** -0.5
        self.use_rope = use_rope

        if use_rope:
            assert head_features % 2 == 0, "head_features must be even for RoPE"
            self.rotary = RotaryEmbedding(head_features)

        mid = num_heads * head_features
        if out_features is None:
            out_features = mid
        self.to_out = nn.Linear(in_features=mid, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # shapes: q: [B, Nq, mid], k: [B, Nk, mid], v: [B, Nk, mid]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        if self.use_rope:
            sin_q, cos_q = self.rotary.get_sin_cos(q.shape[2], device=q.device, dtype=q.dtype)
            sin_k, cos_k = self.rotary.get_sin_cos(k.shape[2], device=k.device, dtype=k.dtype)
            q = apply_rotary_pos_emb(q, sin_q, cos_q)
            k = apply_rotary_pos_emb(k, sin_k, cos_k)

        sim = einsum("b h n d, b h m d -> b h n m", q, k)
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
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_features = head_features
        mid = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.Identity()
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(in_features=features, out_features=mid, bias=False)
        self.to_kv = nn.Linear(in_features=context_features, out_features=mid * 2, bias=False)

        # -> WAŻNA ZMIANA: wymuszamy out_features == features
        self.base = AttentionBase(
            num_heads=num_heads,
            head_features=head_features,
            out_features=features,
            use_rope=use_rope,
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

# --- DiT blocks -------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    DiT-style block: pre-normalization via AdaLNZero before attention and FFN.
    Supports self-attention + optional cross-attention (context).
    """

    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        cond_dim: int,
        context_features: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = exists(context_features) and context_features > 0

        # adaptive layer norms
        self.attn_mod = AdaLNZero(dim=features, cond_dim=cond_dim)
        self.ffn_mod = AdaLNZero(dim=features, cond_dim=cond_dim)

        # self-attention and optional cross-attention (full-KV)
        self.self_attn = Attention(
            features=features,
            head_features=head_features,
            num_heads=num_heads,
            use_rope=use_rope,
        )

        if self.use_cross_attention:
            self.cross_attn = Attention(
                features=features,
                head_features=head_features,
                num_heads=num_heads,
                context_features=context_features,
                use_rope=use_rope,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, cond: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # Self-attention (pre-norm & conditioning)
        attn_in = self.attn_mod(x, cond)
        x = self.self_attn(attn_in) + x

        # Cross-attention (if context provided)
        if self.use_cross_attention and exists(context):
            cross_in = self.attn_mod(x, cond)
            x = self.cross_attn(cross_in, context=context) + x

        # Feed-forward
        ffn_in = self.ffn_mod(x, cond)
        x = self.feed_forward(ffn_in) + x

        return x


# --- Main DiT1d model ------------------------------------------------------

class _BaseDiT1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: int,
        embedding_max_length: int = 512,
        use_style_conditioning: bool = False,
        style_features: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__()
        assert exists(context_embedding_features), "context_embedding_features must be provided"
        self.use_style_conditioning = use_style_conditioning

        token_dim = channels  # token dimensionality (we use cross-attention to context)
        cond_dim = token_dim

        # time positional -> cond_dim
        self.to_time = nn.Sequential(TimePositionalEmbedding(dim=channels, out_features=cond_dim), nn.GELU())

        if use_style_conditioning:
            assert exists(style_features), "style_features must be provided when use_style_conditioning=True"
            self.to_style = nn.Sequential(nn.Linear(in_features=style_features, out_features=cond_dim), nn.GELU())

        self.mapping = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        # build blocks with cross-attention to embeddings
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    features=token_dim,
                    num_heads=num_heads,
                    head_features=head_features,
                    multiplier=multiplier,
                    cond_dim=cond_dim,
                    context_features=context_embedding_features,
                    use_rope=use_rope,
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
        L = embedding.shape[1]
        x_tokens = x.expand(-1, L, -1)  # [B, L, C] - queries
        tokens = x_tokens  # we use cross-attention to conditioning embeddings

        cond = self.get_conditioning(time, features)  # [B, cond_dim]

        for block in self.blocks:
            tokens = block(tokens, cond=cond, context=embedding)

        tokens = self.out_norm(tokens)
        tokens = tokens.mean(dim=1)  # avg pool over seq length -> [B, C]
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


# --- Public wrapper --------------------------------------------------------

class Transformer1d(_BaseDiT1d):
    """API-compatible DiT1d (single unified version with cross-attention & RoPE)."""

    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
        use_rope: bool = True,
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
            use_rope=use_rope,
        )



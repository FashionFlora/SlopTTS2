# file: Modules/diffusion/style_prior.py
# A robust, expressive diffusion prior for style vectors
# Keeps EDM/Karras compatibility so your sampler/schedule work unmodified.

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

# ------------------------------
# small utils
# ------------------------------

def exists(x):
    return x is not None

def default(x, d):
    return x if exists(x) else d() if callable(d) else d

def rand_bool(shape, proba, device):
    return torch.rand(shape, device=device) < proba

# ------------------------------
# Positional time embedding
# ------------------------------

class LearnedTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.w = nn.Parameter(torch.randn(half))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        f = t * self.w[None, :] * 2.0 * math.pi
        fourier = torch.cat([f.sin(), f.cos()], dim=-1)
        # include raw t to help very small t
        return torch.cat([t, fourier], dim=-1)

class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# AdaLN-Zero (DiT-style)
# ------------------------------

class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        # zero init last layer to stabilize early training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [B, T, C], c: [B, cond_dim]
        scale, shift = self.mlp(c).chunk(2, dim=-1)
        x = self.ln(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ------------------------------
# Multi-Head Self/Cross Attention
# ------------------------------

class MHA(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        # x: [B, T, C]; context: [B, S, C] or None -> self-attn
        if context is None:
            context = x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                      (q, k, v))
        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h n d, b h m d -> b h n m", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h n m, b h m d -> b h n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

# ------------------------------
# Style-DiT block: AdaLN-Zero + MHA + FFN
# ------------------------------

class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)

class StyleDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.self_attn = MHA(dim, heads, head_dim, dropout)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.cross_attn = MHA(dim, heads, head_dim, dropout)
        self.adaln3 = AdaLNZero(dim, cond_dim)
        self.ff = FFN(dim, mult=4, dropout=dropout)

    def forward(self, x, cond_vec, context):
        # x: [B, T, C]; context: [B, S, C]
        x = x + self.self_attn(self.adaln1(x, cond_vec))
        x = x + self.cross_attn(self.adaln2(x, cond_vec), context=context)
        x = x + self.ff(self.adaln3(x, cond_vec))
        return x

# ------------------------------
# StyleDiT backbone for style vectors
# ------------------------------

class StyleDiT(nn.Module):
    """
    A compact DiT-like model to predict denoised style vectors.
    We treat the style vector as a length-T sequence (T small; e.g., 1 or few
    learnable "style tokens") to preserve DiT’s inductive bias. Typically T=4
    tokens works well; output is mean pooled to style_dim.
    """
    def __init__(
        self,
        style_dim: int,            # final output dim (e.g., 256 for gs+dur)
        model_dim: int = 512,      # hidden width
        num_tokens: int = 4,       # style tokens
        depth: int = 8,
        heads: int = 8,
        head_dim: int = 64,
        time_embed_dim: int = 128, # time conditioning
        txt_dim: int = 768,        # BERT hidden size
        ref_dim: int = 256,        # reference style feature dim (gs||dur)
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # style token input proj
        self.in_proj = nn.Linear(style_dim, model_dim)

        # learned token embeddings (to let the model expand a single vector)
        self.token_embed = nn.Parameter(
            torch.randn(1, num_tokens, model_dim) * 0.02
        )

        # time embedding
        self.time_pe = LearnedTimeEmbed(time_embed_dim)
        self.time_mlp = TimeMLP(time_embed_dim + 1, model_dim)

        # text conditioning proj
        self.txt_proj = nn.Linear(txt_dim, model_dim)

        # ref conditioning proj (for multi-ref, we will average before proj)
        self.ref_proj = nn.Linear(ref_dim, model_dim)

        # fusion: build a single cond vector (mean over sequence)
        cond_dim = model_dim * 3  # time + pooled text + pooled ref

        self.blocks = nn.ModuleList([
            StyleDiTBlock(
                dim=model_dim,
                heads=heads,
                head_dim=head_dim,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.out_ln = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, style_dim)

        # small zero-init to stabilize early steps
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _cond_vector(
        self,
        t: torch.Tensor,           # [B]
        txt: torch.Tensor,         # [B, L, txt_dim]
        ref: torch.Tensor,         # [B, R, ref_dim] (R refs, concat gs||dur or any)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # time
        t_feat = self.time_pe(t)     # [B, time_embed_dim+1]
        t_feat = self.time_mlp(t_feat)  # [B, model_dim]

        # text: mean pool
        txt_ctx = self.txt_proj(txt)    # [B, L, model_dim]
        txt_pool = txt_ctx.mean(dim=1)  # [B, model_dim]

        # ref: mean pool across refs
        ref_ctx = self.ref_proj(ref)    # [B, R, model_dim]
        ref_pool = ref_ctx.mean(dim=1)  # [B, model_dim]

        cond = torch.cat([t_feat, txt_pool, ref_pool], dim=-1)  # [B, cond_dim]
        # use text as "context" for cross-attn (you can also concat ref_ctx)
        context = txt_ctx
        return cond, context

    def forward(
        self,
        x_noisy: torch.Tensor,     # [B, 1, style_dim] or [B, T, style_dim]
        t: torch.Tensor,           # [B] time in [0, 1] (we derive from sigma)
        txt: torch.Tensor,         # [B, L, txt_dim] (BERT)
        ref: torch.Tensor,         # [B, R, ref_dim]  (gs||dur or variants)
    ) -> torch.Tensor:
        b = x_noisy.size(0)
        # flatten style sequence to tokens
        if x_noisy.dim() == 3:
            tokens_in = x_noisy  # [B, T, style_dim]
        else:
            tokens_in = x_noisy.view(b, 1, -1)  # [B, 1, style_dim]

        # project tokens into model space
        h = self.in_proj(tokens_in)  # [B, T, model_dim]

        # broadcast learned token embeddings and concat
        tok = self.token_embed.expand(b, self.num_tokens, -1)  # [B, T2, C]
        h = torch.cat([h, tok], dim=1)  # [B, T+num_tokens, C]

        # build cond vector + context
        cond_vec, context = self._cond_vector(t, txt, ref)  # [B, Cc], [B, L, C]

        # transformer blocks with AdaLN-Zero and self/cross-attn
        for blk in self.blocks:
            h = blk(h, cond_vec, context)

        h = self.out_ln(h)
        h = h.mean(dim=1)  # pool tokens
        out = self.out_proj(h)  # [B, style_dim] — predicts x_pred in EDM
        return out

# ------------------------------
# Karras EDM with Min-SNR weight + EMA + CFG
# ------------------------------

class KDiffusionWithEMA(nn.Module):
    """
    EDM-style Karras diffusion wrapper with:
    - Min-SNR loss weighting for stability
    - Classifier-free guidance (cfg) for text conditioning
    - EMA of network parameters
    API compatible with your current KDiffusion usage:
      - denoise_fn(x_noisy, sigma or sigmas, ...) returns x_denoised
      - forward(x0, noise=None, ...) -> scalar loss
    """
    alias = "k"

    def __init__(
        self,
        net: nn.Module,                 # StyleDiT
        sigma_data: float = 0.5,
        sigma_min: float = 1e-4,
        sigma_max: float = 3.0,
        min_snr_gamma: float = 5.0,     # Min-SNR weighting
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.min_snr_gamma = min_snr_gamma

        # EMA
        self.ema_decay = ema_decay
        self.net_ema = copy_model(net)

    @torch.no_grad()
    def ema_update(self):
        for p, p_ema in zip(self.net.parameters(), self.net_ema.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    @staticmethod
    def _sigmas_to_t(sigmas: torch.Tensor) -> torch.Tensor:
        # Use VK mapping (t in [0,1]): t = atan(sigma)/(pi/2)
        return torch.atan(sigmas) / (math.pi / 2.0)

    def _get_scale_weights(self, sigmas: torch.Tensor):
        # same as your KDiffusion
        sd = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        s = sigmas.view(-1, 1, 1)
        c_skip = (sd ** 2) / (s ** 2 + sd ** 2)
        c_out = s * sd * (sd ** 2 + s ** 2) ** -0.5
        c_in = (s ** 2 + sd ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: torch.Tensor,            # [B, 1, D] style vector noisy
        sigmas: Optional[torch.Tensor] = None,
        sigma: Optional[float] = None,
        *,
        embedding: torch.Tensor,          # [B, L, txt_dim]
        features: Optional[torch.Tensor] = None,  # [B, R, ref_dim]
        embedding_mask_proba: float = 0.0,
        embedding_scale: float = 1.0,
        use_ema: bool = True,
    ) -> torch.Tensor:
        b, device = x_noisy.size(0), x_noisy.device
        sigmas = default(
            sigmas, lambda: torch.full((b,), float(sigma), device=device)
        )

        # classifier-free guidance — mask a subset of text embeddings
        txt = embedding
        if embedding_mask_proba > 0:
            mask = rand_bool((b, 1, 1), embedding_mask_proba, device)
            txt_fixed = torch.zeros_like(txt)  # unconditional token
            txt_in = torch.where(mask, txt_fixed, txt)
        else:
            txt_in = txt

        # features (reference) — allow None; if none, use zeros
        if features is None:
            features = torch.zeros(b, 1, self.net.ref_proj.in_features,
                                   device=device)

        # Convert sigmas to t in [0, 1]
        t = self._sigmas_to_t(sigmas)  # [B]

        # Choose net or net_ema for denoising
        net = self.net_ema if use_ema else self.net

        # Predict x_pred (EDM parameterization)
        x_pred = net(x_noisy, t, txt_in, features)
        x_pred = x_pred.view(b, 1, -1)

        # Classifier-free guidance scale:
        if embedding_scale != 1.0 and embedding_mask_proba > 0.0:
            # do one more unconditional pass (use all-masked text)
            txt_uncond = torch.zeros_like(txt)
            x_pred_un = net(x_noisy.transpose(1, 2), t, txt_uncond, features)
            x_pred_un = x_pred_un.view(b, 1, -1)
            x_pred = x_pred_un + (x_pred - x_pred_un) * embedding_scale

        # EDM output transform
        # Karras EDM requires we combine with skip/out weights
        c_skip, c_out, c_in, c_noise = self._get_scale_weights(sigmas)
        x_in = c_in * x_noisy
        # "x_pred" here is the predicted clean target in EDM space
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def _loss_weight(self, sigmas: torch.Tensor) -> torch.Tensor:
        # Min-SNR gamma weighting as in "Min-SNR Diffusion Training"
        # w = gamma / (snr + gamma)
        sd = self.sigma_data
        snr = (sd ** 2) / (sigmas ** 2)
        w = self.min_snr_gamma / (snr + self.min_snr_gamma)
        return w

    def forward(
        self,
        x0: torch.Tensor,                      # [B, 1, D] clean style target
        noise: Optional[torch.Tensor] = None,
        *,
        embedding: torch.Tensor,               # [B, L, txt_dim]
        features: Optional[torch.Tensor] = None,  # [B, R, ref_dim]
        sigma_dist_mean: float = -1.2,
        sigma_dist_std: float = 1.2,
        embedding_mask_proba: float = 0.1,
    ) -> torch.Tensor:
        b, device = x0.size(0), x0.device
        # sample sigmas log-normal (EDM)
        sigmas = torch.exp(
            torch.randn(b, device=device) * sigma_dist_std + sigma_dist_mean
        )
        s = sigmas.view(b, 1, 1)

        noise = default(noise, lambda: torch.randn_like(x0))
        x_noisy = x0 + s * noise

        # Convert to EDM param
        c_skip, c_out, c_in, c_noise = self._get_scale_weights(sigmas)
        x_in = c_in * x_noisy

        # predict x_pred with network
        txt = embedding
        if embedding_mask_proba > 0.0:
            mask = rand_bool((b, 1, 1), embedding_mask_proba, device)
            txt = torch.where(mask, torch.zeros_like(txt), txt)

        if features is None:
            features = torch.zeros(
                b, 1, self.net.ref_proj.in_features, device=device
            )

        t = self._sigmas_to_t(sigmas)
        x_pred = self.net(x_in, t, txt, features)
        x_pred = x_pred.view(b, 1, -1)

        # Weighted MSE against clean x0 in EDM space
        x_denoised = c_skip * x_noisy + c_out * x_pred
        loss = F.mse_loss(x_denoised, x0, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        # apply Min-SNR weight
        w = self._loss_weight(sigmas)
        loss = (loss * w).mean()

        # EMA update outside if you want; or here per step
        self.ema_update()
        return loss

def copy_model(model: nn.Module) -> nn.Module:
    import copy
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m
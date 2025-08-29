# Modules/diffusion/style_diffusion_pro.py
# Prettier print width: 80

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(x, d):
    return x if exists(x) else (d() if callable(d) else d)


def rand_bool(shape, p, device):
    if p <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    if p >= 1:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return torch.bernoulli(torch.full(shape, p, device=device)).bool()


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = None
        self.register_buffer("_initted", torch.tensor(0, dtype=torch.uint8))

    def _init(self):
        self.shadow = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        self._initted[...] = 1

    @torch.no_grad()
    def update(self):
        if self._initted.item() == 0:
            self._init()
            return
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, target: nn.Module):
        if self._initted.item() == 0:
            self._init()
        target.load_state_dict(self.shadow, strict=False)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        gamma, beta = torch.chunk(h, 2, dim=-1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleNet(nn.Module):
    """
    Residual MLP with AdaLayerNorm FiLM conditioning and time embedding.
    Predicts residual in style space. Used for both EDM and CM heads.

    Inputs:
      - x: [B, D] style vector (D=256)
      - time_scalar: [B] (either log-sigma for EDM or t in [0,1] for CM)
      - text_emb: [B, T, Ctxt] or [B, Ctxt]
      - ref: [B, Cref] optional (pass full [acoustic|prosody] if multispeaker)
    """

    def __init__(
        self,
        style_dim: int = 256,
        text_dim: int = 768,
        ref_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
        use_ref: bool = True,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.use_ref = use_ref
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.ref_proj = nn.Linear(ref_dim, ref_dim) if use_ref else None
        cond_dim = text_dim + (ref_dim if use_ref else 0)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.inp = nn.Linear(style_dim, hidden_dim)
        self.adaln = AdaLayerNorm(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList(
            [MLPBlock(hidden_dim, mult=4, dropout=dropout) for _ in range(num_layers)]
        )
        self.out = nn.Linear(hidden_dim, style_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def pool_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        return text_emb.mean(dim=1) if text_emb.ndim == 3 else text_emb

    def build_cond(
        self, text_emb: torch.Tensor, ref: Optional[torch.Tensor]
    ) -> torch.Tensor:
        t = self.text_proj(self.pool_text(text_emb))
        if self.use_ref:
            assert exists(ref), "use_ref=True but ref=None"
            r = self.ref_proj(ref)
            c = torch.cat([t, r], dim=-1)
        else:
            c = t
        return self.cond_proj(c)

    def forward(
        self,
        x: torch.Tensor,
        time_scalar: torch.Tensor,  # [B]
        text_emb: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        embedding_mask_proba: float = 0.0,
        embedding_scale: float = 1.0,
        null_text_emb: Optional[torch.Tensor] = None,
        null_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, device = x.size(0), x.device
        mask = rand_bool((b, 1), embedding_mask_proba, device)

        if null_text_emb is None:
            null_text_emb = torch.zeros_like(text_emb)
        if self.use_ref and null_ref is None:
            null_ref = torch.zeros_like(ref)

        if text_emb.ndim == 3:
            text_c = torch.where(mask.unsqueeze(1), null_text_emb, text_emb)
        else:
            text_c = torch.where(mask, null_text_emb, text_emb)
        ref_c = torch.where(mask, null_ref, ref) if self.use_ref else None

        cond = self.build_cond(text_c, ref_c)
        cond_null = self.build_cond(null_text_emb, null_ref)

        tfeat = self.time_mlp(time_scalar.view(-1, 1))
        h = self.inp(x)
        h = torch.cat([h, tfeat], dim=-1)
        h = self.adaln(h, cond)
        for blk in self.blocks:
            h = h + blk(h)
        out = self.out(h)

        if embedding_scale == 1.0:
            return out

        # unconditional branch for CFG
        h_u = self.inp(x)
        h_u = torch.cat([h_u, tfeat], dim=-1)
        h_u = self.adaln(h_u, cond_null)
        for blk in self.blocks:
            h_u = h_u + blk(h_u)
        out_u = self.out(h_u)
        return out_u + (out - out_u) * embedding_scale


@dataclass
class EDMConfig:
    sigma_data: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float = 3.0
    rho: float = 9.0


class StyleDiffusionPro(nn.Module):
    """
    Unified diffusion module for a 256-D style vector:
      - EDM loss with DPM++ 2M sampling
      - Consistency loss (with EMA teacher) for 1-step sampling
      - Light Gaussian prior regularizer
    """

    def __init__(
        self,
        style_dim: int = 256,
        text_dim: int = 768,
        ref_dim: int = 256,
        use_ref: bool = True,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
        edm_cfg: EDMConfig = EDMConfig(),
        lambda_edm: float = 1.0,
        lambda_cm: float = 0.5,
        lambda_prior: float = 1e-4,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.net = StyleNet(
            style_dim=style_dim,
            text_dim=text_dim,
            ref_dim=ref_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_ref=use_ref,
        )
        self.net_teacher = StyleNet(
            style_dim=style_dim,
            text_dim=text_dim,
            ref_dim=ref_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_ref=use_ref,
        )
        self.ema = EMA(self.net, decay=ema_decay)
        self.ema.copy_to(self.net_teacher)

        self.cfg = edm_cfg
        self.lambda_edm = lambda_edm
        self.lambda_cm = lambda_cm
        self.lambda_prior = lambda_prior

        # sigma_data tracker (EMA)
        self.register_buffer("sigma_data", torch.tensor(self.cfg.sigma_data))

    # ---------- EDM parts ----------

    @staticmethod
    def karras_schedule(sigma_min, sigma_max, rho, steps, device):
        rho_inv = 1.0 / rho
        i = torch.arange(steps, device=device, dtype=torch.float32)
        sigmas = (sigma_max**rho_inv + i / (steps - 1) *
                  (sigma_min**rho_inv - sigma_max**rho_inv)) ** rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])], dim=0)

    def _edm_scales(self, sigma: torch.Tensor):
        sd = self.sigma_data
        c_skip = (sd**2) / (sigma**2 + sd**2)
        c_out = sigma * sd * (sd**2 + sigma**2) ** -0.5
        c_in = (sigma**2 + sd**2) ** -0.5
        return c_skip, c_out, c_in

    def denoise_fn(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        embedding_mask_proba: float = 0.0,
        embedding_scale: float = 1.0,
    ) -> torch.Tensor:
        c_skip, c_out, c_in = self._edm_scales(sigma)
        x_in = x_noisy * c_in.view(-1, 1)
        pred = self.net(
            x_in,
            time_scalar=torch.log(sigma + 1e-12),
            text_emb=embedding,
            ref=features,
            embedding_mask_proba=embedding_mask_proba,
            embedding_scale=embedding_scale,
        )
        return c_skip.view(-1, 1) * x_noisy + c_out.view(-1, 1) * pred

    def edm_loss(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        features: Optional[torch.Tensor],
        embedding_mask_proba: float = 0.1,
    ) -> torch.Tensor:
        b, device = x.size(0), x.device
        u = torch.rand(b, device=device)
        sigma = (
            self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            + u * (
                self.cfg.sigma_min ** (1.0 / self.cfg.rho)
                - self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            )
        ) ** self.cfg.rho
        noise = torch.randn_like(x)
        x_noisy = x + sigma.view(-1, 1) * noise

        x_hat = self.denoise_fn(
            x_noisy,
            sigma=sigma,
            embedding=embedding,
            features=features,
            embedding_mask_proba=embedding_mask_proba,
            embedding_scale=1.0,
        )
        sd = self.sigma_data
        w = (sigma**2 + sd**2) * (sigma * sd) ** -2
        loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=1) * w
        return loss.mean()

    # ---------- Consistency parts ----------

    @torch.no_grad()
    def _teacher_x0(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: Optional[torch.Tensor],
    ):
        c_skip, c_out, c_in = self._edm_scales(sigma)
        x_in = x_noisy * c_in.view(-1, 1)
        pred = self.net_teacher(
            x_in,
            time_scalar=torch.log(sigma + 1e-12),
            text_emb=embedding,
            ref=features,
            embedding_mask_proba=0.0,
            embedding_scale=1.0,
        )
        x_hat = c_skip.view(-1, 1) * x_noisy + c_out.view(-1, 1) * pred
        return x_hat

    def consistency_loss(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, device = x.size(0), x.device
        u1 = torch.rand(b, device=device)
        u2 = torch.rand(b, device=device)
        s1 = (
            self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            + u1 * (
                self.cfg.sigma_min ** (1.0 / self.cfg.rho)
                - self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            )
        ) ** self.cfg.rho
        s2 = (
            self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            + u2 * (
                self.cfg.sigma_min ** (1.0 / self.cfg.rho)
                - self.cfg.sigma_max ** (1.0 / self.cfg.rho)
            )
        ) ** self.cfg.rho
        s1, s2 = torch.maximum(s1, s2), torch.minimum(s1, s2)

        eps1 = torch.randn_like(x)
        eps2 = torch.randn_like(x)
        x1 = x + s1.view(-1, 1) * eps1
        x2 = x + s2.view(-1, 1) * eps2

        with torch.no_grad():
            x0_teacher = self._teacher_x0(x1, s1, embedding, features)

        x0_student = self.denoise_fn(
            x2,
            sigma=s2,
            embedding=embedding,
            features=features,
            embedding_mask_proba=0.0,
            embedding_scale=1.0,
        )
        w = (s2 / (s1 + 1e-8)).clamp(min=0.2, max=1.0)
        return (w * F.mse_loss(x0_student, x0_teacher, reduction="none").mean(dim=1)).mean()

    # ---------- Prior regularizer ----------

    def prior_kl(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x**2).mean()

    # ---------- Training forward ----------

    def forward(
        self,
        x: torch.Tensor,  # [B, 256] acoustic-first
        embedding: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        *,
        embedding_mask_proba: float = 0.1,
        update_sigma_data_ema: bool = True,
    ) -> torch.Tensor:
        if update_sigma_data_ema:
            with torch.no_grad():
                sd_batch = x.std(dim=-1).mean()
                self.sigma_data.mul_(0.99).add_(sd_batch, alpha=0.01)

        loss_edm = self.edm_loss(
            x, embedding=embedding, features=features,
            embedding_mask_proba=embedding_mask_proba
        )
        loss_cm = self.consistency_loss(x, embedding=embedding, features=features)
        loss_prior = self.prior_kl(x)

        loss = (
            self.lambda_edm * loss_edm
            + self.lambda_cm * loss_cm
            + self.lambda_prior * loss_prior
        )

        self.ema.update()
        self.ema.copy_to(self.net_teacher)
        return loss

    # ---------- Sampling ----------

    @torch.no_grad()
    def sample_edm(
        self,
        noise: torch.Tensor,  # [B, 256]
        embedding: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        *,
        steps: int = 5,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        embedding_scale: float = 1.0,
    ) -> torch.Tensor:
        device = noise.device
        sigmas = self.karras_schedule(
            self.cfg.sigma_min, self.cfg.sigma_max, self.cfg.rho, steps, device
        )
        x = sigmas[0] * noise
        d_prev = None
        sigma_prev_hat = None
        for i in range(steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]
            gamma = 0.0
            if (sigma_i >= s_tmin) and (sigma_i <= s_tmax):
                gamma = min(s_churn / max(steps - 1, 1), math.sqrt(2.0) - 1.0)
            sigma_hat = sigma_i * (1.0 + gamma)
            if gamma > 0.0:
                eps = torch.randn_like(x) * s_noise
                x = x + (sigma_hat**2 - sigma_i**2).sqrt() * eps

            den = self.denoise_fn(
                x,
                sigma=torch.full((x.size(0),), sigma_hat, device=device),
                embedding=embedding,
                features=features,
                embedding_mask_proba=0.0,
                embedding_scale=embedding_scale,
            )
            d = (x - den) / sigma_hat
            h = sigma_next - sigma_hat
            if d_prev is None:
                x_next = x + h * d
            else:
                h_prev = sigma_hat - sigma_prev_hat
                r = h / (h_prev + 1e-12)
                x_next = x + h * ((1 + r) * d - r * d_prev)
            d_prev = d
            sigma_prev_hat = sigma_hat
            x = x_next
        return x

    @torch.no_grad()
    def sample_cm(
        self,
        noise: torch.Tensor,  # [B, 256]
        embedding: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        *,
        steps: int = 1,
        embedding_scale: float = 1.0,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        x = noise
        if steps <= 1:
            sigma = torch.full((x.size(0),), self.cfg.sigma_max, device=x.device)
            model = self.net_teacher if use_teacher else self.net
            c_skip, c_out, c_in = self._edm_scales(sigma)
            x_in = x * c_in.view(-1, 1)
            pred = model(
                x_in,
                time_scalar=torch.log(sigma + 1e-12),
                text_emb=embedding,
                ref=features,
                embedding_mask_proba=0.0,
                embedding_scale=embedding_scale,
            )
            x0 = c_skip.view(-1, 1) * x + c_out.view(-1, 1) * pred
            return x0

        ts = torch.linspace(self.cfg.sigma_max, 0.0, steps + 1, device=x.device)
        for i in range(steps):
            s = ts[i]
            s_next = ts[i + 1]
            den = self.denoise_fn(
                x,
                sigma=torch.full((x.size(0),), s, device=x.device),
                embedding=embedding,
                features=features,
                embedding_mask_proba=0.0,
                embedding_scale=embedding_scale,
            )
            d = (x - den) / (s + 1e-8)
            h = s_next - s
            x = x + h * d
        return x
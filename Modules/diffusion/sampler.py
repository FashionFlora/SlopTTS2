from math import atan, cos, pi, sin, sqrt
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops import reduce as einops_reduce
from .utils import *

"""
Diffusion Training
"""

""" Distributions """


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class UniformDistribution(Distribution):
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.rand(num_samples, device=device)





""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs


class Diffusion(nn.Module):

    alias: str = ""

    """Base diffusion class"""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError("Diffusion class missing denoise_fn")

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        raise NotImplementedError("Diffusion class missing forward function")


class VDiffusion(Diffusion):

    alias = "v"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Get noise
        noise = default(noise, lambda: torch.randn_like(x))

        # Combine input and noise weighted by half-circle
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta

        # Denoise and return loss
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        return F.mse_loss(x_denoised, x_target)


class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device
        from einops import rearrange, reduce

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise
        
        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = einops_reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss


class VKDiffusion(nn.Module):
    alias = "vk"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        min_snr_gamma: Optional[float] = 5.0,
        robust: Optional[str] = "huber",   # None | "huber" | "charbonnier"
        huber_delta: float = 0.5,
        charbonnier_eps: float = 1e-3,
        sigma_min_clamp: float = 1e-3,
        sigma_max_clamp: float = 1.0,
        normalize_weight: bool = True,
        lambda_x0: float = 0.1,            # NOWE: waga kotwicy x0
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.min_snr_gamma = min_snr_gamma
        self.robust = robust
        self.huber_delta = huber_delta
        self.charbonnier_eps = charbonnier_eps
        self.sigma_min_clamp = sigma_min_clamp
        self.sigma_max_clamp = sigma_max_clamp
        self.normalize_weight = normalize_weight
        self.lambda_x0 = lambda_x0

    def get_scale_weights(self, sigmas: Tensor):
        sigmas = sigmas.clamp(self.sigma_min_clamp, self.sigma_max_clamp)
        sigmas_ = rearrange(sigmas, "b -> b 1 1")
        denom = (sigmas_**2 + 1.0)
        c_skip = 1.0 / denom
        c_out  = -sigmas_ * (denom**-0.5)
        c_in   = denom**-0.5
        return c_skip, c_out, c_in, sigmas

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        sigmas = sigmas.clamp(self.sigma_min_clamp, self.sigma_max_clamp)
        return sigmas.atan() / pi * 2.0

    @torch.no_grad()
    def denoise_fn(self, x_noisy: Tensor, sigmas: Optional[Tensor] = None, sigma: Optional[float] = None, **kwargs) -> Tensor:
        b, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(batch_size=b, device=device, x=sigma, xs=sigmas)
        c_skip, c_out, c_in, _ = self.get_scale_weights(sigmas)
        t = self.sigma_to_t(sigmas)
        x_pred = self.net(c_in * x_noisy, t, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor = None, return_aux: bool = False, **kwargs):
        b, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=b, device=device)
        c_skip, c_out, c_in, sigmas = self.get_scale_weights(sigmas)

        x_noisy = x + rearrange(sigmas, "b -> b 1 1") * default(noise, lambda: torch.randn_like(x))
        t = self.sigma_to_t(sigmas)
        x_pred = self.net(c_in * x_noisy, t, **kwargs)  # v-pred

        # v loss
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-8)
        if self.robust == "huber":
            losses_v = F.smooth_l1_loss(x_pred, v_target, reduction="none", beta=self.huber_delta)
        elif self.robust == "charbonnier":
            diff = x_pred - v_target
            losses_v = torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps)
        else:
            losses_v = F.mse_loss(x_pred, v_target, reduction="none")
        per_ex_v = einops_reduce(losses_v, "b ... -> b", "mean")

        # x0 consistency (kotwica)
        x_hat = c_skip * x_noisy + c_out * x_pred     # rekonstrukcja x
        losses_x0 = F.l1_loss(x_hat, x, reduction="none")
        per_ex_x0 = einops_reduce(losses_x0, "b ... -> b", "mean")

        # Min-SNR wagi (na v-loss; można też na x0, ale zwykle nie trzeba)
        if self.min_snr_gamma is not None:
            snr = 1.0 / (sigmas**2 + 1e-8)
            w = snr.clamp(max=self.min_snr_gamma) / (snr + 1e-8)
            if self.normalize_weight:
                w = w / (w.mean().detach() + 1e-8)
            per_ex_v = per_ex_v * w

        loss = per_ex_v.mean() + self.lambda_x0 * per_ex_x0.mean()

        if return_aux:
            return loss, {
                "loss_v": per_ex_v.mean().detach(),
                "loss_x0": per_ex_x0.mean().detach(),
                "sigma_med": sigmas.median().detach(),
            }
        return loss

"""
Diffusion Sampling
"""

""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def forward(self, num_steps: int, device: Any) -> Tensor:
        sigmas = torch.linspace(1, 0, num_steps + 1)[:-1]
        return sigmas


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


""" Samplers """


class Sampler(nn.Module):

    diffusion_types: List[Type[Diffusion]] = []

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")








""" Main Classes """


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
        clamp: bool = True,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs}),
                     sigmas=sigmas, num_steps=sigmas.numel())
        return x


class DiffusionInpainter(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        num_resamples: int,
        sampler: Sampler,
        sigma_schedule: Schedule,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor) -> Tensor:
        x = self.inpaint_fn(
            source=inpaint,
            mask=inpaint_mask,
            fn=self.denoise_fn,
            sigmas=self.sigma_schedule(self.num_steps, inpaint.device),
            num_steps=self.num_steps,
            num_resamples=self.num_resamples,
        )
        return x


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):
    def __init__(
        self,
        inpainter: DiffusionInpainter,
        *,
        num_spans: int,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool = False) -> Tensor:
        half_length = start.shape[2] // 2

        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        # Inpaint second half from first half
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)

        for i in range(self.num_spans):
            # Inpaint second half
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            # Replace first half with generated second half
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=2)

class DPMpp2MSampler(Sampler):
    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, s_churn: float = 0.0, s_tmin: float = 0.0,
                 s_tmax: float = float("inf"), s_noise: float = 1.0):
        super().__init__()
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def step(
        self,
        x: Tensor,
        fn: Callable,
        sigma: float,
        sigma_next: float,
        i: int,
        n: int,
        d_prev: Optional[Tensor],
        sigma_prev_hat: Optional[float],
    ) -> Tuple[Tensor, Tensor, float]:
        # Churn (EDM trick)
        gamma = 0.0
        if self.s_tmin <= sigma <= self.s_tmax:
            gamma = min(self.s_churn / max(n - 1, 1), sqrt(2.0) - 1.0)
        sigma_hat = sigma * (1.0 + gamma)
        if gamma > 0.0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + (sigma_hat**2 - sigma**2).sqrt() * eps

        # Current derivative
        denoised = fn(x, sigma=sigma_hat)
        d = (x - denoised) / sigma_hat
        h = sigma_next - sigma_hat

        if d_prev is None:
            # First step -> Euler
            x_next = x + h * d
        else:
            # 2M multistep
            # h_prev is the last effective step size
            h_prev = sigma_hat - sigma_prev_hat
            r = h / (h_prev + 1e-12)
            x_next = x + h * ((1 + r) * d - r * d_prev)

        return x_next, d, sigma_hat

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        d_prev: Optional[Tensor] = None
        sigma_prev_hat: Optional[float] = None

        for i in range(num_steps - 1):
            x, d_prev, sigma_prev_hat = self.step(
                x,
                fn=fn,
                sigma=sigmas[i],
                sigma_next=sigmas[i + 1],
                i=i,
                n=num_steps,
                d_prev=d_prev,
                sigma_prev_hat=sigma_prev_hat,
            )
        return x

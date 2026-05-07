"""DDPM forward process + DDPM/DDIM samplers.

Conventions:
    alphas_bar = cumulative product of (1 - betas)
    Forward:    q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    Reverse:    epsilon-prediction.

Cosine schedule per Nichol & Dhariwal 2021.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch


def cosine_betas(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
    f = torch.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clip(betas, 0.0, max_beta).float()


def linear_betas(num_timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)


class DDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine") -> None:
        self.num_timesteps = num_timesteps
        if schedule == "cosine":
            betas = cosine_betas(num_timesteps)
        elif schedule == "linear":
            betas = linear_betas(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        # posterior variance for DDPM
        self.posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    def to(self, device: torch.device) -> "DDPMScheduler":
        for name in (
            "betas",
            "alphas",
            "alphas_bar",
            "alphas_bar_prev",
            "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ── Training helpers ──────────────────────────────────────────────────
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sm = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        return a * x0 + sm * noise, noise

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch,), device=device, dtype=torch.long)

    # ── DDPM ancestral sampling ───────────────────────────────────────────
    @torch.no_grad()
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x_t - coef * eps_pred) / torch.sqrt(alpha_t)
        if t == 0:
            return mean
        var = self.posterior_variance[t]
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    # ── DDIM deterministic sampling ───────────────────────────────────────
    def get_ddim_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        step = max(1, self.num_timesteps // num_inference_steps)
        ts = list(range(0, self.num_timesteps, step))[:num_inference_steps]
        return torch.tensor(ts[::-1], dtype=torch.long)

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        a_t = self.alphas_bar[t]
        a_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t)) * torch.sqrt(1.0 - a_t / a_prev)
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0)) * eps_pred
        out = torch.sqrt(a_prev) * x0_pred + dir_xt
        if eta > 0 and t_prev >= 0:
            out = out + sigma * torch.randn_like(x_t)
        return out


@torch.no_grad()
def ddpm_sample(
    model,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    y: torch.Tensor | None,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Full DDPM ancestral sampling (slow but high quality).

    Args:
        cond: (B, C_cond, H, W) tensor concatenated to noise input each step.
        y:    (B,) class labels or None.
        shape: shape of noise input — (B, 1, H, W) for L-only target.
    """
    x = torch.randn(shape, device=device)
    null = (
        torch.full((shape[0],), model.num_classes, device=device, dtype=torch.long)
        if y is not None and guidance_scale != 1.0
        else None
    )
    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_in = torch.cat([x, cond], dim=1)
        eps = model(x_in, t_batch, y)
        if null is not None:
            eps_uncond = model(x_in, t_batch, null)
            eps = eps_uncond + guidance_scale * (eps - eps_uncond)
        x = scheduler.ddpm_step(x, eps, t)
    return x


@torch.no_grad()
def ddim_sample(
    model,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    y: torch.Tensor | None,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """DDIM sampling. 50 steps usually suffices for our task."""
    x = torch.randn(shape, device=device)
    timesteps = scheduler.get_ddim_timesteps(num_steps).tolist()
    null = (
        torch.full((shape[0],), model.num_classes, device=device, dtype=torch.long)
        if y is not None and guidance_scale != 1.0
        else None
    )
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_in = torch.cat([x, cond], dim=1)
        eps = model(x_in, t_batch, y)
        if null is not None:
            eps_uncond = model(x_in, t_batch, null)
            eps = eps_uncond + guidance_scale * (eps - eps_uncond)
        x = scheduler.ddim_step(x, eps, t, t_prev, eta=eta)
    return x

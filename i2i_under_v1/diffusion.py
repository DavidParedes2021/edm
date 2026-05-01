"""
Cosine-schedule DDPM with DDIM sampling and Classifier-Free Guidance.

Lightweight self-contained implementation rather than depending on the
diffusers 0.14 scheduler API (which is a moving target). All buffers are
registered on the module so they migrate to the right device automatically
with `.to(device)` -- no more alphas_cumprod CPU/CUDA mismatches.
"""
from __future__ import annotations
from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn


def cosine_betas(num_steps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule (Improved DDPM)."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    alphas_bar = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5).pow(2)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=max_beta)


class DDPMScheduler(nn.Module):
    """Buffers for forward noising; methods for q_sample, predict_x0, DDIM step."""

    def __init__(self, num_train_timesteps: int = 1000, schedule: str = 'cosine'):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        if schedule == 'cosine':
            betas = cosine_betas(num_train_timesteps)
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alphas_cumprod', alphas_cumprod, persistent=False)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt(), persistent=False)

    @staticmethod
    def _gather(buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = buf.gather(0, t.long()).float()
        return out.view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_t = sqrt(a_bar) x0 + sqrt(1-a_bar) noise."""
        s_a = self._gather(self.sqrt_alphas_cumprod, t, x0.ndim)
        s_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return s_a * x0 + s_1ma * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Estimate x0 from x_t and predicted noise."""
        s_a = self._gather(self.sqrt_alphas_cumprod, t, x_t.ndim)
        s_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        return (x_t - s_1ma * eps) / s_a.clamp(min=1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
        x_init: Optional[torch.Tensor] = None,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampler.

        eps_fn(x_t, t) -> predicted noise. Wrap CFG and conditioning inside.
        x_init: if provided, start from this (used for SDEdit-style sampling
                downstream); otherwise pure Gaussian.
        """
        # Evenly-spaced subset of training timesteps, descending.
        ts = torch.linspace(self.num_train_timesteps - 1, 0, num_steps + 1, device=device).long()
        x = x_init if x_init is not None else torch.randn(shape, device=device)
        for i in range(num_steps):
            t = ts[i].expand(shape[0])
            t_prev = ts[i + 1].expand(shape[0])
            ab_t = self._gather(self.alphas_cumprod, t, x.ndim)
            ab_p = self._gather(self.alphas_cumprod, t_prev, x.ndim).clamp(min=1e-8)

            eps = eps_fn(x, t)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
            sigma = eta * ((1 - ab_p) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_p).sqrt()
            dir_xt = (1 - ab_p - sigma ** 2).clamp(min=0.0).sqrt() * eps
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = ab_p.sqrt() * x0 + dir_xt + sigma * noise
        return x


def cfg_eps_fn(
    model: nn.Module,
    y_cond: torch.Tensor,
    cls_target: torch.Tensor,
    null_cls: int,
    cfg_scale: float,
):
    """
    Build an `eps_fn(x_t, t) -> eps` closure that applies Classifier-Free
    Guidance over the class label.
    """
    def fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if cfg_scale == 1.0:
            return model(x_t, y_cond, t, cls_target)
        # Two forward passes batched together.
        x2 = torch.cat([x_t, x_t], dim=0)
        c2 = torch.cat([y_cond, y_cond], dim=0)
        t2 = torch.cat([t, t], dim=0)
        cls_null = torch.full_like(cls_target, null_cls)
        cls2 = torch.cat([cls_target, cls_null], dim=0)
        eps_both = model(x2, c2, t2, cls2)
        eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    return fn


# ---- self-test --------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    sched = DDPMScheduler(num_train_timesteps=1000)
    print(f"alphas_cumprod[0]={sched.alphas_cumprod[0]:.4f}, "
          f"alphas_cumprod[-1]={sched.alphas_cumprod[-1]:.4e}")
    x0 = torch.randn(2, 1, 16, 16)
    t = torch.tensor([10, 500])
    n = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, n)
    x0_hat = sched.predict_x0(xt, t, n)
    err = (x0 - x0_hat).abs().max().item()
    print(f"x0 round-trip err: {err:.2e}")
    assert err < 1e-4
    print("diffusion.py OK")

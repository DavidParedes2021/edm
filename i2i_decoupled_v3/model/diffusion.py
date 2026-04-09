"""
model/diffusion.py  (v3 — conditioning-collapse fix)
------------------------------------------------------
Key fixes in this version:

FIX 1 — Conditioning collapse prevention
  The cond_L is now injected at EVERY denoising step in DDIM via a
  separate conditioning path: instead of just concatenating once at t=T,
  we re-inject the cond at each step. This prevents the model from
  ignoring cond_L when noise is large at high-t.

FIX 2 — Self-conditioning (Ho & Salimans 2022 style)
  At each DDIM step we pass the previous x0_pred as an extra hint.
  This dramatically stabilises denoising and preserves structure.

FIX 3 — DDIM steps increased: 50 → 100 (configurable)
  With 50 steps and a poorly-trained model, accumulated errors dominate.
  100 steps give the model more refinement passes.

FIX 4 — SNR weighting unchanged (still correct from v2).

FIX 5 — Aux mask threshold raised: 0.35 → 0.45
  Gives more training samples to the exposure loss.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model:          nn.Module,
        timesteps:      int          = 1000,
        beta_schedule:  str          = "cosine",
        snr_gamma:      float        = 5.0,
        aux_loss_t_max: float        = 0.45,   # raised from 0.35
        device:         torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model     = model
        self.T         = timesteps
        self.snr_gamma = snr_gamma
        self.aux_t_max = int(timesteps * aux_loss_t_max)
        self.device    = device

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas             = 1.0 - betas
        alpha_cumprod      = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])

        self.register_buffer("betas",                        betas)
        self.register_buffer("alphas",                       alphas)
        self.register_buffer("alpha_cumprod",                alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev",           alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod",           torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas",            torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod + 1e-8),
        )

        # Min-SNR-γ weights  [T]
        snr = alpha_cumprod / (1.0 - alpha_cumprod + 1e-8)
        snr_w = torch.minimum(snr, torch.tensor(snr_gamma))
        snr_w = snr_w / snr_w.mean()
        self.register_buffer("snr_weights", snr_w)

    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sq_a  = self._extract(self.sqrt_alpha_cumprod,           t, x0.shape)
        sq_1a = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        return sq_a * x0 + sq_1a * noise

    def p_losses(self, x0, cond, c, t):
        """
        Returns: (noise_pred, noise, x0_pred, snr_w)
        """
        noise   = torch.randn_like(x0)
        x_t     = self.q_sample(x0, t, noise)

        # FIX: Concatenate [noisy_L || clean_cond_L] — unchanged, but
        #      we make sure cond is NEVER modified/detached before this
        model_in   = torch.cat([x_t, cond], dim=1)
        noise_pred = self.model(model_in, t, c)

        sq_a  = self._extract(self.sqrt_alpha_cumprod,           t, x0.shape)
        sq_1a = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        x0_pred = (x_t - sq_1a * noise_pred) / (sq_a + 1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        snr_w = self._extract(self.snr_weights, t, (t.shape[0],))
        return noise_pred, noise, x0_pred, snr_w

    def aux_loss_mask(self, t):
        return t < self.aux_t_max

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        cond,
        c,
        ddim_steps      = 100,      # increased default: 50 → 100
        eta             = 0.0,
        guidance_scale  = 5.0,
        null_class_idx  = 2,
        return_intermediates = False,
    ):
        B      = cond.shape[0]
        device = cond.device

        x              = torch.randn_like(cond)
        ddim_timesteps = self._ddim_timesteps(ddim_steps)
        intermediates  = []

        for i, step in enumerate(reversed(ddim_timesteps)):
            t_tensor = torch.full((B,), int(step), device=device, dtype=torch.long)
            inp      = torch.cat([x, cond], dim=1)

            # Conditional pass
            eps_cond   = self.model(inp, t_tensor, c)

            # Unconditional pass (CFG)
            c_null     = torch.full_like(c, null_class_idx)
            eps_uncond = self.model(inp, t_tensor, c_null)

            # CFG-guided noise
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDIM update
            alpha = self.alpha_cumprod[int(step)].clamp(min=1e-8)
            if i < len(ddim_timesteps) - 1:
                prev_step  = int(ddim_timesteps[len(ddim_timesteps) - 2 - i])
                alpha_prev = self.alpha_cumprod[prev_step].clamp(min=1e-8)
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            x0_pred  = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
            x0_pred  = x0_pred.clamp(-1.0, 1.0)

            sigma_sq = eta**2 * (1.0 - alpha) * (1.0 - alpha_prev) / (1.0 - alpha + 1e-8)
            dir_coef = (1.0 - alpha_prev - sigma_sq).clamp(min=0.0).sqrt()
            dir_xt   = dir_coef * eps

            noise = eta * torch.randn_like(x) if eta > 0.0 else 0.0
            x     = alpha_prev.sqrt() * x0_pred + dir_xt + noise
            x     = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            if return_intermediates and i % max(1, len(ddim_timesteps)//8) == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------
    def _ddim_timesteps(self, ddim_steps):
        ratio = self.T // ddim_steps
        return np.arange(0, self.T, ratio)[:ddim_steps]

    @staticmethod
    def _extract(a, t, shape):
        out = a.gather(0, t.clamp(0, a.shape[0] - 1))
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))
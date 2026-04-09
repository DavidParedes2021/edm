"""
model/diffusion.py  (v2 - fixed)
----------------------------------
Fixes applied vs v1:
  1. Min-SNR-γ weighted diffusion loss (Hang et al. 2023)
     Upweights low-t steps where image content is recoverable.
     Downweights high-t (pure noise) steps that dominated training v1.
  2. Auxiliary losses (exposure, structure) are gated to low-t steps only.
     At high-t, x0_pred is obtained by dividing by a tiny sqrt(alpha),
     which saturates after clamping, producing zero gradients.
  3. Numerically stable DDIM update step.
  4. Biased timestep sampling: aux losses use t ~ U(0, T//3).
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


# ---------------------------------------------------------------------------
# Gaussian Diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model:          nn.Module,
        timesteps:      int            = 1000,
        beta_schedule:  str            = "cosine",
        snr_gamma:      float          = 5.0,   # Min-SNR-γ clamp value
        aux_loss_t_max: float          = 0.35,  # Apply aux losses only for t < T*this fraction
        device:         torch.device   = torch.device("cpu"),
    ):
        super().__init__()
        self.model          = model
        self.T              = timesteps
        self.snr_gamma      = snr_gamma
        self.aux_t_max      = int(timesteps * aux_loss_t_max)  # e.g. 350 for T=1000
        self.device         = device

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

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

        # Precompute Min-SNR-γ loss weights  [T]
        snr = alpha_cumprod / (1.0 - alpha_cumprod + 1e-8)
        snr_weights = torch.minimum(snr, torch.tensor(snr_gamma))
        # Normalise so mean weight ≈ 1 (keeps LR in the same ballpark as unweighted)
        snr_weights = snr_weights / snr_weights.mean()
        self.register_buffer("snr_weights", snr_weights)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sq_a  = self._extract(self.sqrt_alpha_cumprod,           t, x0.shape)
        sq_1a = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        return sq_a * x0 + sq_1a * noise

    # ------------------------------------------------------------------
    # Training  — returns everything the trainer needs
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x0:   torch.Tensor,   # [B,1,H,W] clean L_target
        cond: torch.Tensor,   # [B,1,H,W] clean L_normal
        c:    torch.Tensor,   # [B] class labels (with CFG null already applied)
        t:    torch.Tensor,   # [B] timesteps
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        noise_pred   [B,1,H,W]  — predicted noise
        noise        [B,1,H,W]  — actual noise added
        x0_pred      [B,1,H,W]  — x0 estimated from noise_pred (valid only at low t)
        snr_w        [B]         — Min-SNR-γ per-sample weights
        """
        noise   = torch.randn_like(x0)
        x_t     = self.q_sample(x0, t, noise)

        model_in   = torch.cat([x_t, cond], dim=1)   # [B,2,H,W]
        noise_pred = self.model(model_in, t, c)

        # Reconstruct x0 from predicted noise
        sq_a  = self._extract(self.sqrt_alpha_cumprod,           t, x0.shape)
        sq_1a = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        x0_pred = (x_t - sq_1a * noise_pred) / (sq_a + 1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Per-sample SNR weights
        snr_w = self._extract(self.snr_weights, t, (t.shape[0],))  # [B]

        return noise_pred, noise, x0_pred, snr_w

    def aux_loss_mask(self, t: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True for samples where t < aux_t_max (low noise, valid x0_pred)."""
        return t < self.aux_t_max   # [B]

    # ------------------------------------------------------------------
    # DDIM sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        cond:                 torch.Tensor,
        c:                    torch.Tensor,
        ddim_steps:           int   = 50,
        eta:                  float = 0.0,
        guidance_scale:       float = 4.0,
        null_class_idx:       int   = 2,
        return_intermediates: bool  = False,
    ) -> torch.Tensor:
        B      = cond.shape[0]
        device = cond.device

        x              = torch.randn_like(cond)
        ddim_timesteps = self._ddim_timesteps(ddim_steps)
        intermediates  = []

        for i, step in enumerate(reversed(ddim_timesteps)):
            t_tensor = torch.full((B,), int(step), device=device, dtype=torch.long)

            inp        = torch.cat([x, cond], dim=1)
            eps_cond   = self.model(inp, t_tensor, c)
            c_null     = torch.full_like(c, null_class_idx)
            eps_uncond = self.model(inp, t_tensor, c_null)

            # Classifier-Free Guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDIM update — numerically stable
            alpha = self.alpha_cumprod[int(step)].clamp(min=1e-8)
            if i < len(ddim_timesteps) - 1:
                prev_step  = int(ddim_timesteps[len(ddim_timesteps) - 2 - i])
                alpha_prev = self.alpha_cumprod[prev_step].clamp(min=1e-8)
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # Predicted clean image
            x0_pred = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM direction
            sigma_sq = eta ** 2 * (1.0 - alpha) * (1.0 - alpha_prev) / (1.0 - alpha + 1e-8)
            dir_coef = (1.0 - alpha_prev - sigma_sq).clamp(min=0.0).sqrt()
            dir_xt   = dir_coef * eps

            noise = eta * torch.randn_like(x) if eta > 0.0 else 0.0
            x     = alpha_prev.sqrt() * x0_pred + dir_xt + noise
            x     = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            if return_intermediates and i % (max(1, len(ddim_timesteps) // 8)) == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------
    def _ddim_timesteps(self, ddim_steps: int) -> np.ndarray:
        ratio = self.T // ddim_steps
        return np.arange(0, self.T, ratio)[:ddim_steps]

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        B   = t.shape[0]
        out = a.gather(0, t.clamp(0, a.shape[0] - 1))
        return out.reshape(B, *([1] * (len(shape) - 1)))
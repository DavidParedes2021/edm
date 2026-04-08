"""
model/diffusion.py
-------------------
DDPM training process + DDIM inference sampler.

Key decisions
-------------
* Cosine noise schedule (Nichol & Dhariwal 2021):  avoids the degenerate
  early and late timesteps of the linear schedule, producing significantly
  sharper samples because less noise is injected near t=0.

* Predict the noise (eps-parametrisation): standard, stable, works with
  classifier-free guidance (CFG).

* DDIM sampler (Song et al. 2020): deterministic (eta=0), produces sharp
  outputs in 50 steps instead of 1000.  Blurriness in naive DDPM comes from
  averaging over many stochastic paths; DDIM avoids this.

* Classifier-Free Guidance (CFG): at training, 15% of the time the class
  label is replaced with a special "null" token (num_classes index).  At
  inference, the noise prediction is nudged away from unconditional toward
  the target class:
      eps = eps_uncond + guidance_scale * (eps_cond − eps_uncond)
  This is the primary mechanism that enforces *strong* exposure changes.
  scale ≥ 3.0 produces large illumination shifts.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal 2021).
    Returns beta tensor of shape [T].
    """
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
    """
    Wraps the UNet with DDPM forward process and DDIM reverse sampler.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int      = 1000,
        beta_schedule: str  = "cosine",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model     = model
        self.T         = timesteps
        self.device    = device

        # ── Noise schedule ──────────────────────────────────────────────
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas      = 1.0 - betas
        alpha_cumprod      = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])

        # Register buffers so they move with .to(device)
        self.register_buffer("betas",               betas)
        self.register_buffer("alphas",              alphas)
        self.register_buffer("alpha_cumprod",       alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev",  alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod",  torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("log_one_minus_alpha_cumprod",  torch.log(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas",   torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sq_a  = self._extract(self.sqrt_alpha_cumprod,      t, x0.shape)
        sq_1a = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        return sq_a * x0 + sq_1a * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x0:    torch.Tensor,        # [B, 1, H, W] clean L_target
        cond:  torch.Tensor,        # [B, 1, H, W] clean L_normal (spatial cond)
        c:     torch.Tensor,        # [B] class label
        t:     torch.Tensor,        # [B] sampled timestep
    ) -> torch.Tensor:
        """Return predicted noise (used to compute all losses in training.py)."""
        noise  = torch.randn_like(x0)
        x_t    = self.q_sample(x0, t, noise)

        # Concatenate spatial conditioning: [noisy_L, cond_L] → 2 channels
        model_input = torch.cat([x_t, cond], dim=1)   # [B, 2, H, W]

        noise_pred = self.model(model_input, t, c)
        return noise_pred, noise

    # ------------------------------------------------------------------
    # DDIM inference sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        cond:               torch.Tensor,   # [B, 1, H, W] L_normal
        c:                  torch.Tensor,   # [B] class label
        ddim_steps:         int    = 50,
        eta:                float  = 0.0,   # 0 = deterministic
        guidance_scale:     float  = 4.0,
        null_class_idx:     int    = 2,     # index of the null (unconditional) class
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling with Classifier-Free Guidance.

        Returns denoised L_target of shape [B, 1, H, W].
        """
        B = cond.shape[0]
        device = cond.device

        # Start from pure noise
        x = torch.randn_like(cond).to(device)

        # Select DDIM timestep subsequence
        ddim_timesteps = self._ddim_timesteps(ddim_steps)
        intermediates  = []

        for i, step in enumerate(reversed(ddim_timesteps)):
            t_tensor = torch.full((B,), step, device=device, dtype=torch.long)

            # ── Conditional prediction ───────────────────────────────
            inp      = torch.cat([x, cond], dim=1)
            eps_cond = self.model(inp, t_tensor, c)

            # ── Unconditional prediction (null class) ────────────────
            c_null   = torch.full_like(c, null_class_idx)
            eps_uncond = self.model(inp, t_tensor, c_null)

            # ── CFG combination ──────────────────────────────────────
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # ── DDIM update step ─────────────────────────────────────
            alpha      = self.alpha_cumprod[step].clamp(min=1e-6)
            alpha_prev = (
                self.alpha_cumprod[ddim_timesteps[len(ddim_timesteps) - 2 - i]]
                if i < len(ddim_timesteps) - 1
                else torch.tensor(1.0, device=device)
            ).clamp(min=1e-6)

            # Predicted x0
            x0_pred = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Direction pointing to x_t (numerically stable)
            sigma_sq = eta ** 2 * (1.0 - alpha) * (1.0 - alpha_prev) / (1.0 - alpha + 1e-8)
            dir_coef = (1.0 - alpha_prev - sigma_sq).clamp(min=0.0).sqrt()
            dir_xt   = dir_coef * eps

            noise = eta * torch.randn_like(x) if eta > 0 else 0.0
            x     = alpha_prev.sqrt() * x0_pred + dir_xt + noise
            # Guard against NaN from untrained / edge-case models
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            if return_intermediates and i % 10 == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ddim_timesteps(self, ddim_steps: int) -> np.ndarray:
        """Select evenly-spaced subset of the T timesteps."""
        ratio    = self.T // ddim_steps
        timesteps = np.arange(0, self.T, ratio)[:ddim_steps]
        return timesteps

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather values from 'a' at timestep indices 't', broadcast to 'shape'."""
        B    = t.shape[0]
        out  = a.gather(-1, t)
        return out.reshape(B, *([1] * (len(shape) - 1)))

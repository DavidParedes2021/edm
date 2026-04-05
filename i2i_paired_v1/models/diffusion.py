"""
models/diffusion.py
DDPM noise scheduler + DDIM sampler with v-prediction support.

v-prediction (Salimans & Ho 2022):
  v_t = √ᾱ_t · ε - √(1-ᾱ_t) · x_0
  x_0 = √ᾱ_t · x_t - √(1-ᾱ_t) · v

This formulation yields sharper samples vs. pure ε-prediction.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class DDPMScheduler:
    """
    Linear-beta DDPM noise schedule with optional v-prediction.

    Args:
        num_timesteps    : T (e.g. 1000)
        beta_start       : β_1 (e.g. 1e-4)
        beta_end         : β_T (e.g. 0.02)
        prediction_type  : "epsilon" or "v_prediction"
    """

    def __init__(
        self,
        num_timesteps: int   = 1000,
        beta_start: float    = 1e-4,
        beta_end: float      = 0.02,
        prediction_type: str = "v_prediction",
    ):
        self.num_timesteps   = num_timesteps
        self.prediction_type = prediction_type

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        self.alphas        = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # ᾱ_t

        # Useful pre-computed values
        self.sqrt_alphas_cumprod        = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For v-prediction: √(1-ᾱ) / √ᾱ
        self.sqrt_recip_alphas_cumprod  = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # DDPM posterior q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64),
                                         self.alphas_cumprod[:-1]])
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    # ── Device helpers ────────────────────────────────────────────────

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather values at timestep indices t and broadcast to shape."""
        device = t.device
        arr    = arr.to(device)
        out    = arr[t.long()].float()
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    # ── Forward process  q(x_t | x_0) ────────────────────────────────

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t given x_0, ε, t."""
        sqrt_acp   = self._extract(self.sqrt_alphas_cumprod,             t, x0.shape)
        sqrt_1macp = self._extract(self.sqrt_one_minus_alphas_cumprod,   t, x0.shape)
        return sqrt_acp * x0 + sqrt_1macp * noise

    # ── Prediction conversions ────────────────────────────────────────

    def get_velocity(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute v_t = √ᾱ_t · ε - √(1-ᾱ_t) · x_0"""
        sqrt_acp   = self._extract(self.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1macp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * noise - sqrt_1macp * x0

    def predict_x0_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_0 = √ᾱ_t · x_t - √(1-ᾱ_t) · v"""
        sqrt_acp   = self._extract(self.sqrt_alphas_cumprod,           t, xt.shape)
        sqrt_1macp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return sqrt_acp * xt - sqrt_1macp * v

    def predict_x0_from_eps(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_0 = (x_t - √(1-ᾱ) · ε) / √ᾱ"""
        sqrt_recip  = self._extract(self.sqrt_recip_alphas_cumprod,   t, xt.shape)
        sqrt_recipm1= self._extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape)
        return sqrt_recip * xt - sqrt_recipm1 * eps

    def predict_x0(self, xt: torch.Tensor, model_out: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict x_0 using the configured prediction type."""
        if self.prediction_type == "v_prediction":
            return self.predict_x0_from_v(xt, model_out, t)
        else:  # epsilon
            return self.predict_x0_from_eps(xt, model_out, t)

    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get the training target given the prediction type."""
        if self.prediction_type == "v_prediction":
            return self.get_velocity(x0, noise, t)
        else:
            return noise

    # ── DDPM reverse step  p(x_{t-1} | x_t) ─────────────────────────

    @torch.no_grad()
    def ddpm_step(
        self,
        model_out: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Single DDPM reverse step → x_{t-1}"""
        x0_pred = self.predict_x0(xt, model_out, t)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean_c1  = self._extract(self.posterior_mean_coef1, t, xt.shape)
        mean_c2  = self._extract(self.posterior_mean_coef2, t, xt.shape)
        mean     = mean_c1 * x0_pred + mean_c2 * xt

        log_var  = self._extract(self.posterior_log_variance_clipped, t, xt.shape)
        noise    = torch.randn_like(xt)
        # No noise at t=0
        nonzero  = (t > 0).float().view(-1, *([1] * (xt.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise


# ──────────────────────────────────────────────────────────────────────────────
# DDIM sampler (deterministic, fast)
# ──────────────────────────────────────────────────────────────────────────────

class DDIMSampler:
    """
    DDIM sampling (Song et al., 2020).
    Deterministic reverse process – much sharper than DDPM at 50 steps.

    Args:
        scheduler        : DDPMScheduler instance (provides betas/alphas)
        num_inference_steps: number of DDIM steps (≤ T)
        eta              : 0 = deterministic, 1 = DDPM
    """

    def __init__(
        self,
        scheduler: DDPMScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.scheduler           = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta                 = eta

        T = scheduler.num_timesteps
        # Sub-sample timesteps uniformly
        step_ratio = T // num_inference_steps
        self.timesteps = list(reversed(range(0, T, step_ratio)))[:num_inference_steps]

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        exposure: torch.Tensor,
        device: torch.device,
        clip_denoised: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            model    : IlluminationUNet
            shape    : (B, 3, H, W) – batch shape of output
            cond     : [B, 3, H, W] normal frame condition
            exposure : [B] float exposure labels
            device   : target device
        Returns:
            x0 : [B, 3, H, W] denoised (generated artifact)
        """
        sched    = self.scheduler
        acp      = sched.alphas_cumprod.to(device)
        timesteps = self.timesteps

        # Start from pure Gaussian noise
        xt = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

            model_out = model(xt, t_batch, cond, exposure)

            # Predicted x0
            x0_pred = sched.predict_x0(xt, model_out, t_batch)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Predicted noise
            acp_t   = acp[t_val]
            eps_pred = (xt - acp_t.sqrt() * x0_pred) / (1 - acp_t).sqrt().clamp(min=1e-8)

            # Next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                acp_prev = acp[t_prev]
            else:
                acp_prev = torch.tensor(1.0, device=device)

            sigma_t = self.eta * torch.sqrt(
                (1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)
            )
            noise = torch.randn_like(xt) if self.eta > 0 else torch.zeros_like(xt)

            xt = (
                acp_prev.sqrt() * x0_pred
                + torch.sqrt(1 - acp_prev - sigma_t**2) * eps_pred
                + sigma_t * noise
            )

            if verbose:
                print(f"  DDIM step {i+1}/{len(timesteps)}: t={t_val}")

        return xt


# ──────────────────────────────────────────────────────────────────────────────
# Helper to build scheduler from config
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(cfg: dict) -> DDPMScheduler:
    model_cfg = cfg["model"]
    return DDPMScheduler(
        num_timesteps   = model_cfg["timesteps"],
        prediction_type = model_cfg.get("prediction_type", "v_prediction"),
    )


def build_sampler(cfg: dict, scheduler: DDPMScheduler) -> DDIMSampler:
    inf_cfg = cfg["inference"]
    return DDIMSampler(
        scheduler           = scheduler,
        num_inference_steps = inf_cfg["num_inference_steps"],
        eta                 = inf_cfg.get("ddim_eta", 0.0),
    )


if __name__ == "__main__":
    sched = DDPMScheduler(num_timesteps=100, prediction_type="v_prediction")
    x0    = torch.randn(2, 3, 64, 64)
    noise = torch.randn_like(x0)
    t     = torch.tensor([50, 10])
    xt    = sched.add_noise(x0, noise, t)
    v     = sched.get_velocity(x0, noise, t)
    x0_r  = sched.predict_x0_from_v(xt, v, t)
    err   = (x0 - x0_r).abs().max().item()
    print(f"Reconstruction error (should be ~0): {err:.6f}")

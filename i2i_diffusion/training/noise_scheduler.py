"""
training/noise_scheduler.py
---------------------------
Thin wrapper around diffusers 0.14's DDPMScheduler and DDIMScheduler.

diffusers 0.14 API notes
-------------------------
- DDPMScheduler.__init__ accepts:
    num_train_timesteps, beta_start, beta_end, beta_schedule,
    clip_sample, prediction_type  (added in 0.11)
- DDPMScheduler.add_noise(original, noise, timesteps) → noisy_sample
- DDIMScheduler.step(model_output, timestep, sample) → DDIMSchedulerOutput
    .prev_sample is the denoised estimate
- Scheduler.alphas_cumprod is the cumulative product ᾱ_t  [T]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from diffusers import DDIMScheduler, DDPMScheduler


class DiffusionScheduler:
    """
    Unified interface for training (DDPM) and inference (DDIM).

    Parameters
    ----------
    num_train_timesteps : int
    beta_schedule       : str   "linear" or "cosine"
    beta_start, beta_end: float
    prediction_type     : str   "epsilon" (predict noise) or "v_prediction"
    clip_sample         : bool
    """

    def __init__(
        self,
        num_train_timesteps: int   = 1000,
        beta_schedule:       str   = "linear",
        beta_start:          float = 1e-4,
        beta_end:            float = 2e-2,
        prediction_type:     str   = "epsilon",
        clip_sample:         bool  = False,
    ) -> None:
        shared_kwargs = dict(
            num_train_timesteps = num_train_timesteps,
            beta_schedule       = beta_schedule,
            beta_start          = beta_start,
            beta_end            = beta_end,
            clip_sample         = clip_sample,
        )

        self.train_scheduler = DDPMScheduler(
            **shared_kwargs,
            prediction_type = prediction_type,
        )

        self.infer_scheduler = DDIMScheduler(
            **shared_kwargs,
            prediction_type = prediction_type,
        )

        self.num_train_timesteps = num_train_timesteps

    # ── training helpers ──────────────────────────────────────────────────────

    def sample_timesteps(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> torch.Tensor:
        """Uniform random timesteps in [0, T-1]."""
        return torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        """
        Move all scheduler CPU tensors (betas, alphas, alphas_cumprod, etc.)
        to `device`.  diffusers 0.14 schedulers are plain Python objects whose
        tensor attributes live on CPU by default; calling this method keeps
        them co-located with the model so add_noise / step never see a
        cross-device index.
        """
        for attr in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        ):
            for sched in (self.train_scheduler, self.infer_scheduler):
                val = getattr(sched, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(sched, attr, val.to(device))
        return self

    def add_noise(
        self,
        clean:     torch.Tensor,   # (B, C, H, W)
        noise:     torch.Tensor,   # (B, C, H, W)  sampled ~ N(0, I)
        timesteps: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0).
        Ensures scheduler buffers are on the same device as inputs.
        """
        # Keep alphas_cumprod co-located with the input — cheap if already there
        ac = self.train_scheduler.alphas_cumprod
        if ac.device != clean.device:
            self.to(clean.device)
        return self.train_scheduler.add_noise(clean, noise, timesteps)

    def get_noise_target(
        self,
        clean:     torch.Tensor,
        noise:     torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns what the model should predict given (noisy, t).
        For prediction_type='epsilon', this is the noise itself.
        """
        # diffusers 0.14 DDPMScheduler does not expose a direct
        # get_velocity / get_target method, so we handle both cases:
        pt = self.train_scheduler.config.get("prediction_type", "epsilon")
        if pt == "epsilon":
            return noise
        # v-prediction target
        alphas_cumprod = self.train_scheduler.alphas_cumprod.to(clean.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_om_alpha   = (1 - alphas_cumprod[timesteps]) ** 0.5
        # reshape for broadcasting
        sqap = sqrt_alpha_prod[:, None, None, None]
        somp = sqrt_om_alpha[:, None, None, None]
        return sqap * noise - somp * clean

    # ── inference helpers ─────────────────────────────────────────────────────

    def set_inference_timesteps(self, num_steps: int = 50) -> None:
        self.infer_scheduler.set_timesteps(num_steps)

    @property
    def inference_timesteps(self) -> torch.Tensor:
        return self.infer_scheduler.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep:     int | torch.Tensor,
        sample:       torch.Tensor,
    ) -> torch.Tensor:
        """Single DDIM denoising step.  Returns prev_sample.
        Moves infer scheduler buffers to sample's device if needed.
        """
        ac = self.infer_scheduler.alphas_cumprod
        if ac.device != sample.device:
            self.to(sample.device)
        out = self.infer_scheduler.step(model_output, timestep, sample)
        return out.prev_sample
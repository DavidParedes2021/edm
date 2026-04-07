# utils/diffusion.py
"""
DDPM / DDIM scheduler wrapper compatible with diffusers==0.14.0.
Handles:
  - Noise scheduling (forward process)
  - Classifier-Free Guidance (CFG) during inference
  - Safe timestep sampling
"""
import torch
from diffusers import DDPMScheduler, DDIMScheduler


def build_scheduler(
    num_train_timesteps: int = 1000,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    clip_sample: bool = True,
) -> DDPMScheduler:
    """
    Build DDPM training scheduler.
    squaredcos_cap_v2 (cosine schedule) is smoother than linear and
    produces better results for image translation tasks.
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
    )
    return scheduler


def build_inference_scheduler(
    num_inference_steps: int = 50,
    num_train_timesteps: int = 1000,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
) -> DDIMScheduler:
    """
    Build DDIM inference scheduler (faster, deterministic).
    50 steps ≈ quality of 1000-step DDPM.
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=True,
    )
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def add_noise(
    scheduler: DDPMScheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Apply forward diffusion: q(x_t | x_0) = sqrt(α_t)*x_0 + sqrt(1-α_t)*ε
    All tensors must be on the same device (validated here).
    """
    assert clean.device == noise.device, \
        f"clean on {clean.device}, noise on {noise.device}"

    # DDPMScheduler.add_noise keeps everything on clean's device
    noisy = scheduler.add_noise(clean, noise, timesteps)
    return noisy


def sample_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample random timesteps for training batch."""
    return torch.randint(
        0,
        num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def ddim_sample(
    model,
    scheduler: DDIMScheduler,
    shape: tuple,
    exposure_labels: torch.Tensor,
    condition_images: torch.Tensor,
    device: torch.device,
    cfg_scale: float = 7.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Full DDIM reverse diffusion with Classifier-Free Guidance.

    Args:
        model:            trained UNet
        scheduler:        DDIM scheduler with set_timesteps already called
        shape:            (B, 1, H, W) — luminance channel shape
        exposure_labels:  (B,) long tensor — 0=over, 1=under
        condition_images: (B, 1, H, W) — normal frame Y channel (conditioning)
        device:           target device
        cfg_scale:        guidance strength (5–7 recommended)
        dtype:            float32 or float16 (AMP)

    Returns:
        y_generated: (B, 1, H, W) luminance in [-1, 1]
    """
    # Start from pure Gaussian noise
    x = torch.randn(shape, device=device, dtype=dtype)

    # Null condition token (all-zeros label = uncond)
    null_labels = torch.full_like(exposure_labels, fill_value=2)  # 2 = null class

    model.eval()
    for t in scheduler.timesteps:
        t_batch = t.expand(shape[0]).to(device)

        # Conditional prediction
        noise_pred_cond = model(
            x,
            t_batch,
            exposure_label=exposure_labels,
            condition_y=condition_images,
        )

        # Unconditional prediction (CFG)
        noise_pred_uncond = model(
            x,
            t_batch,
            exposure_label=null_labels,
            condition_y=condition_images,
        )

        # CFG formula: ε = ε_uncond + γ * (ε_cond - ε_uncond)
        noise_pred = noise_pred_uncond + cfg_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # DDIM step — compatible with diffusers 0.14.0 API
        scheduler_out = scheduler.step(
            noise_pred,
            t,
            x,
        )
        x = scheduler_out.prev_sample

    model.train()
    return x

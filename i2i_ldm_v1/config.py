"""
config.py — Central configuration for the Illumination Diffusion pipeline.

Automatically adapts batch size, resolution and precision based on available
VRAM so the same file runs on a 4 GB laptop GPU (smoke-test) and a 16 GB DGX
GPU (full training) without editing anything.
"""

import torch


def get_device_profile() -> dict:
    """
    Detect VRAM and return a profile dict that drives every memory-sensitive
    hyper-parameter.  Four tiers:
        • smoke  (<= 4 GB, e.g. RTX 3050 laptop)
        • low    (<= 8 GB)
        • mid    (<= 16 GB, e.g. one DGX GPU slice)
        • high   (> 16 GB)
    """
    if not torch.cuda.is_available():
        return _profile("cpu")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if vram_gb <= 4.5:
        return _profile("smoke")
    elif vram_gb <= 8.5:
        return _profile("low")
    elif vram_gb <= 16.5:
        return _profile("mid")
    else:
        return _profile("high")


def _profile(tier: str) -> dict:
    profiles = {
        "cpu": dict(
            tier=tier,
            image_size=64,
            train_batch=1,
            grad_accum=1,
            mixed_precision="no",        # fp32 on CPU
            vae_encode_batch=1,
            unet_channels=(128, 256, 384, 512),
            unet_layers_per_block=1,
            latent_channels=4,
            save_every=50,
            sample_every=50,
            max_train_steps=200,          # just verify loop
            gradient_checkpointing=False,
            dataloader_workers=0,
        ),
        "smoke": dict(
            tier=tier,
            image_size=128,
            train_batch=1,
            grad_accum=2,
            mixed_precision="fp16",
            vae_encode_batch=1,
            unet_channels=(128, 256, 384, 512),
            unet_layers_per_block=1,
            latent_channels=4,
            save_every=100,
            sample_every=100,
            max_train_steps=500,          # smoke-test: verify convergence starts
            gradient_checkpointing=True,
            dataloader_workers=2,
        ),
        "low": dict(
            tier=tier,
            image_size=256,
            train_batch=2,
            grad_accum=4,
            mixed_precision="fp16",
            vae_encode_batch=2,
            unet_channels=(128, 256, 384, 512),
            unet_layers_per_block=2,
            latent_channels=4,
            save_every=500,
            sample_every=500,
            max_train_steps=20_000,
            gradient_checkpointing=True,
            dataloader_workers=4,
        ),
        "mid": dict(
            tier=tier,
            image_size=512,
            train_batch=2,
            grad_accum=4,                 # effective batch = 8
            mixed_precision="fp16",
            vae_encode_batch=4,
            unet_channels=(256, 512, 768, 1024),
            unet_layers_per_block=2,
            latent_channels=4,
            save_every=1_000,
            sample_every=1_000,
            max_train_steps=50_000,
            gradient_checkpointing=True,
            dataloader_workers=4,
        ),
        "high": dict(
            tier=tier,
            image_size=512,
            train_batch=8,
            grad_accum=1,
            mixed_precision="bf16",
            vae_encode_batch=8,
            unet_channels=(256, 512, 768, 1024),
            unet_layers_per_block=2,
            latent_channels=4,
            save_every=2_000,
            sample_every=2_000,
            max_train_steps=100_000,
            gradient_checkpointing=False,  # enough VRAM — faster without it
            dataloader_workers=8,
        ),
    }
    return profiles[tier]


# ──────────────────────────────────────────────────────────────────────────────
# Static config — shared across all tiers
# ──────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    # Paths
    data_dir_normal = "../../../data/datasets/ead_2020_classified/edm2020_classified/normal_frames"
    data_dir_over   = "../../../data/datasets/ead_2020_classified/edm2020_classified/overexposed_frames"
    data_dir_under  = "../../../data/datasets/ead_2020_classified/edm2020_classified/underexposed_frames"
    output_dir      = "../../../projects/i2i_ldm_v1/checkpoints"
    samples_dir     = "../../../projects/i2i_ldm_v1/samples"
    log_dir         = "../../../projects/i2i_ldm_v1/logs"

    # Pretrained VAE (SD 1.5 VAE — frozen, used only for encode/decode)
    # Set to None to skip downloading and use a simple pixel-space fallback
    vae_model_id    = "stabilityai/sd-vae-ft-mse"

    # Diffusion schedule
    num_train_timesteps = 1_000
    beta_schedule       = "scaled_linear"   # better than linear for LDM
    beta_start          = 0.00085
    beta_end            = 0.012
    ddim_inference_steps = 50

    # EV conditioning
    # Continuous EV scalar range (stops). Over: positive, Under: negative.
    ev_over_min  =  1.5
    ev_over_max  =  3.0
    ev_under_min = -3.0
    ev_under_max = -1.5
    ev_embed_dim = 256           # sinusoidal EV embedding dimension

    # Loss weights
    lambda_lpips   = 0.8         # perceptual loss — most important add-on
    lambda_ssim    = 0.2         # SSIM on luminance channel
    lambda_cycle   = 0.5         # cycle-consistency (only if USE_CYCLE=True)
    lambda_hist    = 0.1         # histogram loss (only if USE_HIST=True)

    USE_LPIPS      = True
    USE_SSIM       = True
    USE_CYCLE      = False       # enable once base model is stable
    USE_HIST       = False       # enable when you have pseudo-pairs

    # Optimiser
    learning_rate  = 1e-4
    lr_warmup_steps = 500
    adam_beta1     = 0.9
    adam_beta2     = 0.999
    adam_eps       = 1e-8
    adam_weight_decay = 1e-2
    max_grad_norm  = 1.0

    # W&B (set USE_WANDB=False to disable)
    USE_WANDB      = True
    wandb_project  = "illumination-diffusion"
    wandb_run_name = "ldm-ev-conditioning"

    # Reproducibility
    seed           = 42

    # ── derive device-adaptive settings at runtime ──
    def __init__(self):
        profile = get_device_profile()
        for k, v in profile.items():
            setattr(self, k, v)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Config] Device tier: {self.tier} | "
              f"image_size={self.image_size} | "
              f"batch={self.train_batch} | "
              f"mixed_precision={self.mixed_precision}")

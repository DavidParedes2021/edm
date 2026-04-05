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
            mixed_precision="no",
            vae_encode_batch=1,
            unet_channels=(128, 256, 384, 512),
            unet_layers_per_block=1,
            latent_channels=4,
            save_every=50,
            sample_every=50,
            max_train_steps=200,
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
            max_train_steps=500,
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
            grad_accum=4,
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
            gradient_checkpointing=False,
            dataloader_workers=8,
        ),
    }
    return profiles[tier]


# ──────────────────────────────────────────────────────────────────────────────
# Static config — shared across all tiers
# ──────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    # Paths
    #data_dir_normal = "../../../data/datasets/ead_2020_classified/edm2020_classified/normal_frames"
    #data_dir_over   = "../../../data/datasets/ead_2020_classified/edm2020_classified/overexposed_frames"
    #data_dir_under  = "../../../data/datasets/ead_2020_classified/edm2020_classified/underexposed_frames"

    data_dir_normal = "../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames"
    data_dir_over   = "../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/overexposed_frames"
    data_dir_under  = "../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/underexposed_frames"

    output_dir      = "../../../projects/i2i_ldm_v2/checkpoints"
    samples_dir     = "../../../projects/i2i_ldm_v2/samples"
    log_dir         = "../../../projects/i2i_ldm_v2/logs"

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
    ev_over_min  =  1.5
    ev_over_max  =  3.0
    ev_under_min = -3.0
    ev_under_max = -1.5
    ev_embed_dim = 256

    # ── Classifier-Free Guidance (CFG) ──────────────────────────────────────
    # During training, EV conditioning is randomly dropped with probability
    # cfg_dropout and replaced with the null embedding.  At inference the model
    # is run twice (conditioned + unconditioned) and the predictions are
    # extrapolated: pred = uncond + guidance_scale * (cond - uncond).
    #
    # guidance_scale=1.0  → no guidance (equivalent to disabled CFG)
    # guidance_scale=3-7  → recommended range for exposure generation
    # guidance_scale>10   → strong conditioning, risk of mode collapse
    cfg_dropout_prob = 0.10    # fraction of training steps using null cond
    guidance_scale   = 5.0    # used only at inference; set in inference.py

    # ── EMA ─────────────────────────────────────────────────────────────────
    # Exponential Moving Average of the UNet parameters.  EMA weights are
    # more stable and produce sharper outputs at inference.
    use_ema          = True
    ema_decay        = 0.9999

    # ── Min-SNR loss weighting ───────────────────────────────────────────────
    # Clips per-sample diffusion loss weight at γ / SNR(t).
    # Prevents high-noise timesteps from dominating training.
    # gamma=5 is the default from the Min-SNR paper.
    use_snr_weighting = True
    snr_gamma         = 5.0

    # ── Auxiliary loss weights and gating ────────────────────────────────────
    lambda_lpips   = 0.8
    lambda_ssim    = 0.2
    lambda_chroma  = 0.5   # chrominance consistency (new — fixes color shifts)
    lambda_cycle   = 0.5   # disabled by default; enable once base is stable
    lambda_hist    = 0.1
    lambda_exposure = 0.3  # brightness direction loss

    USE_LPIPS      = True
    USE_SSIM       = True
    USE_CHROMA     = True  # new: penalise hue/saturation shifts vs normal
    USE_CYCLE      = False
    USE_HIST       = False
    USE_EXPOSURE   = True

    # Auxiliary losses are only meaningful when x0 is estimated from
    # low-to-mid noise steps (otherwise x0 is dominated by noise).
    # Only apply aux losses when the diffusion timestep t < aux_loss_t_max.
    aux_loss_t_max = 600   # skip aux losses for t >= this threshold

    # Optimiser
    learning_rate    = 1e-4
    lr_warmup_steps  = 500
    adam_beta1       = 0.9
    adam_beta2       = 0.999
    adam_eps         = 1e-8
    adam_weight_decay = 1e-2
    max_grad_norm    = 1.0

    # W&B
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
              f"mixed_precision={self.mixed_precision} | "
              f"CFG dropout={self.cfg_dropout_prob} | "
              f"EMA={'on' if self.use_ema else 'off'}")

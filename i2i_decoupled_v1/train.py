#!/usr/bin/env python3
# train.py
"""
Main training script for Luminance Diffusion Illumination Pipeline.

Usage:
  # Verify on laptop (4GB GPU):
  python train.py --config configs/laptop_debug.yaml

  # Full training on DGX (16GB GPU):
  python train.py --config configs/dgx_train.yaml

  # Resume from last checkpoint (overwrites every save_every steps):
  python train.py --config configs/dgx_train.yaml --resume last

  # Resume from best checkpoint (best val metric so far):
  python train.py --config configs/dgx_train.yaml --resume best

  # Resume from an explicit directory:
  python train.py --config configs/dgx_train.yaml --resume ./outputs/illumination_ycldi_v1/checkpoint-last/
"""
import os
import sys
import argparse
import yaml
import random
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CRITICAL: Set memory config BEFORE any CUDA calls
# ---------------------------------------------------------------------------
def _set_memory_config(max_split_size_mb: int):
    val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if not val:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"


# ---------------------------------------------------------------------------
# Internal imports (after sys.path setup)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from utils.device import get_device, to_device, log_gpu_memory, empty_cache
from utils.color_space import (
    normalize_luminance, denormalize_luminance, replace_luminance
)
from utils.diffusion import build_scheduler, sample_timesteps, add_noise, ddim_sample
from utils.losses import IlluminationDiffusionLoss
from models.unet import IlluminationUNet
from models.ema import EMAModel
from data.dataset import build_dataloader
from evaluation.metrics import evaluate_batch
from diffusers import DDIMScheduler as _DDIMScheduler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# x0 reconstruction from predicted noise
# ---------------------------------------------------------------------------

def predict_x0(
    noisy: torch.Tensor,
    pred_noise: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Reconstruct x_0 from noisy x_t and predicted noise ε_θ:
      x_0 = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)

    This lets us apply pixel-space losses (perceptual, histogram)
    on the reconstructed clean image during training.

    NOTE: alphas_cumprod stays on CPU (it is a scheduler buffer).
          We index it with a CPU copy of timesteps, then move the
          result to device — this avoids the cross-device indexing error:
          "Expected all tensors to be on the same device"
    """
    # alphas_cumprod is always on CPU; index with CPU timesteps
    t_cpu = timesteps.cpu()
    sqrt_alphas    = alphas_cumprod[t_cpu].sqrt().to(device)       # (B,) on device
    sqrt_one_minus = (1.0 - alphas_cumprod[t_cpu]).sqrt().to(device)  # (B,) on device

    # Reshape for broadcasting: (B,) → (B, 1, 1, 1)
    sqrt_alphas    = sqrt_alphas[:, None, None, None]
    sqrt_one_minus = sqrt_one_minus[:, None, None, None]

    x0 = (noisy - sqrt_one_minus * pred_noise) / (sqrt_alphas + 1e-8)
    return x0.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_sample_grid(
    model: nn.Module,
    scheduler,
    val_batch: dict,
    device: torch.device,
    save_path: Path,
    cfg_scale: float = 5.0,
    num_inference_steps: int = 20,
    dtype: torch.dtype = torch.float32,
):
    """Generate and save a grid: rows=samples, cols=[Normal | Over | Under]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    # Take up to 2 samples — guard against val batches smaller than 2
    n_samples = min(2, val_batch["normal_y_norm"].shape[0])
    normal_y_norm = val_batch["normal_y_norm"][:n_samples].to(device, dtype=dtype)
    normal_rgb    = val_batch["normal_rgb"][:n_samples].to(device)

    # Build a fresh DDIM scheduler for inference
    ddim = _DDIMScheduler(
        num_train_timesteps = scheduler.config.num_train_timesteps,
        beta_schedule       = scheduler.config.beta_schedule,
        prediction_type     = scheduler.config.prediction_type,
        clip_sample         = True,
    )
    ddim.set_timesteps(num_inference_steps)

    use_amp = dtype == torch.float16
    results = {}
    for label_val, label_name in [(0, "over"), (1, "under")]:
        labels = torch.full((n_samples,), label_val, dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y_gen = ddim_sample(
                model            = model,
                scheduler        = ddim,
                shape            = normal_y_norm.shape,
                exposure_labels  = labels,
                condition_images = normal_y_norm,
                device           = device,
                cfg_scale        = cfg_scale,
                dtype            = dtype,
            )
        y_gen_01 = denormalize_luminance(y_gen)
        rgb_out  = replace_luminance(normal_rgb.float(), y_gen_01.float())
        results[label_name] = rgb_out.cpu()

    # Plot grid: n_samples rows × 3 cols
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]  # ensure 2-D indexing works for 1 sample
    titles = ["Normal", "Generated Over", "Generated Under"]

    for row in range(n_samples):
        imgs = [
            normal_rgb[row].cpu().permute(1, 2, 0).clamp(0, 1).numpy(),
            results["over"][row].permute(1, 2, 0).clamp(0, 1).numpy(),
            results["under"][row].permute(1, 2, 0).clamp(0, 1).numpy(),
        ]
        for col, (img, title) in enumerate(zip(imgs, titles)):
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sample grid → {save_path}")

    model.train()


# ---------------------------------------------------------------------------
# Checkpoint Manager — keeps only "last" and "best" on disk
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Saves exactly two checkpoints at all times:
      - checkpoint-last/   → overwritten every save_every steps (for resume)
      - checkpoint-best/   → overwritten only when val metric improves

    The metric used for "best" is exposure_visibility_score (higher = better).
    Falls back to histogram KL divergence (lower = better) if EVS is not available.
    No old numbered directories accumulate on disk.
    """

    LAST_DIR = "checkpoint-last"
    BEST_DIR = "checkpoint-best"

    def __init__(self, out_dir: Path, metric: str = "exposure_visibility"):
        self.out_dir     = out_dir
        self.metric      = metric          # val metric key tracked for "best"
        self.best_value  = None            # None = not yet set
        # higher_is_better depends on the metric
        self.higher_is_better = metric != "hist_kl_div"

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.higher_is_better:
            return value > self.best_value
        return value < self.best_value

    def _build_payload(self, model, optimizer, lr_scheduler, scaler,
                       ema, global_step, epoch, cfg) -> dict:
        payload = {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "global_step":  global_step,
            "epoch":        epoch,
            "config":       cfg,
            "best_value":   self.best_value,
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        return payload

    def _save_to(self, directory: str, payload: dict):
        ckpt_dir = self.out_dir / directory
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first, then rename — atomic on Linux/POSIX
        tmp_path  = ckpt_dir / "checkpoint.tmp"
        final_path = ckpt_dir / "checkpoint.pt"
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)

    def save_last(self, model, optimizer, lr_scheduler, scaler,
                  ema, global_step, epoch, cfg):
        """Always overwrite the 'last' checkpoint."""
        payload = self._build_payload(
            model, optimizer, lr_scheduler, scaler, ema, global_step, epoch, cfg
        )
        self._save_to(self.LAST_DIR, payload)
        logger.info(f"[Checkpoint] last → {self.out_dir / self.LAST_DIR} (step {global_step})")

    def save_best_if_improved(self, model, optimizer, lr_scheduler, scaler,
                               ema, global_step, epoch, cfg,
                               val_metrics: dict) -> bool:
        """
        Overwrite 'best' checkpoint only if the tracked metric improved.
        Returns True if the best was updated.
        """
        value = val_metrics.get(self.metric)
        if value is None:
            return False

        if self._is_better(value):
            self.best_value = value
            payload = self._build_payload(
                model, optimizer, lr_scheduler, scaler, ema, global_step, epoch, cfg
            )
            payload["best_value"] = self.best_value
            self._save_to(self.BEST_DIR, payload)
            logger.info(
                f"[Checkpoint] best → {self.out_dir / self.BEST_DIR} "
                f"(step {global_step} | {self.metric}={value:.4f})"
            )
            return True
        return False

    def load(self, which: str, device: torch.device) -> dict:
        """Load 'last' or 'best' checkpoint. which ∈ {'last', 'best'}."""
        dir_map = {"last": self.LAST_DIR, "best": self.BEST_DIR}
        path = self.out_dir / dir_map[which] / "checkpoint.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: Optional[str] = None):
    # 0. Memory config (must be before CUDA init)
    _set_memory_config(cfg.get("memory", {}).get("max_split_size_mb", 512))

    # 1. Setup
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory("startup")

    # Mixed precision dtype
    use_amp = cfg["training"].get("mixed_precision", "no") == "fp16"
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"Mixed precision: {'fp16 AMP' if use_amp else 'fp32'}")

    # Output dirs — all driven from config output section
    ocfg      = cfg.get("output", {})
    base_dir  = Path(ocfg.get("base_dir", f"./outputs/{cfg['experiment_name']}"))
    ckpt_dir  = base_dir / ocfg.get("checkpoint_dir", "checkpoints")
    sample_dir = base_dir / ocfg.get("samples_dir", "samples")
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output base:   {base_dir}")
    logger.info(f"Checkpoints:   {ckpt_dir}")
    logger.info(f"Samples:       {sample_dir}")

    # Checkpoint manager — only "last" and "best" are kept on disk
    ckpt_manager = CheckpointManager(
        out_dir=ckpt_dir,
        metric=cfg.get("training", {}).get("best_metric", "exposure_visibility"),
    )

    # 2. Data — paths come directly from config
    dcfg = cfg["data"]
    train_loader = build_dataloader(
        normal_path=dcfg["normal_path"],
        over_path=dcfg["over_path"],
        under_path=dcfg["under_path"],
        image_size=dcfg["image_size"],
        batch_size=cfg["training"]["batch_size"],
        split="train",
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=dcfg.get("pin_memory", True),
        seed=cfg["seed"],
    )
    val_loader = build_dataloader(
        normal_path=dcfg["normal_path"],
        over_path=dcfg["over_path"],
        under_path=dcfg["under_path"],
        image_size=dcfg["image_size"],
        batch_size=min(4, cfg["training"]["batch_size"]),
        split="val",
        num_workers=2,
        pin_memory=False,
        seed=cfg["seed"],
    )
    # Grab a fixed val batch for visualization
    val_batch_fixed = next(iter(val_loader))

    # 3. Model
    mcfg = cfg["model"]
    model = IlluminationUNet(
        base_channels        = mcfg["base_channels"],
        channel_multipliers  = tuple(mcfg["channel_multipliers"]),
        attention_resolutions= tuple(mcfg["attention_resolutions"]),
        num_res_blocks       = mcfg["num_res_blocks"],
        dropout              = mcfg["dropout"],
        exposure_embed_dim   = mcfg["exposure_embed_dim"],
        image_size           = dcfg["image_size"],
        use_checkpoint       = cfg.get("memory", {}).get("enable_gradient_checkpointing", True),
    ).to(device)

    n_params = model.count_parameters()
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    log_gpu_memory("after model init")

    # EMA
    use_ema = cfg["training"].get("use_ema", False)
    ema = None
    if use_ema:
        ema = EMAModel(model, decay=cfg["training"].get("ema_decay", 0.9999))
        ema.to(device)
        logger.info("EMA enabled")

    # 4. Diffusion scheduler
    difcfg = cfg["diffusion"]
    scheduler = build_scheduler(
        num_train_timesteps = difcfg["num_train_timesteps"],
        beta_schedule       = difcfg["beta_schedule"],
        prediction_type     = difcfg["prediction_type"],
    )
    # alphas_cumprod for x0 reconstruction — keep on CPU, index by timestep
    alphas_cumprod = scheduler.alphas_cumprod  # (T,) on CPU

    # Val DDIM scheduler — built once, reused every epoch (10 steps = fast metric pass)
    ddim_val = _DDIMScheduler(
        num_train_timesteps = difcfg["num_train_timesteps"],
        beta_schedule       = difcfg["beta_schedule"],
        prediction_type     = difcfg["prediction_type"],
        clip_sample         = True,
    )
    ddim_val.set_timesteps(10)

    # 5. Loss
    lcfg = cfg["loss"]
    criterion = IlluminationDiffusionLoss(
        device            = device,
        l1_weight         = lcfg["l1_weight"],
        perceptual_weight = lcfg["perceptual_weight"],
        gradient_weight   = lcfg["gradient_weight"],
        histogram_weight  = lcfg["histogram_weight"],
    )
    criterion = criterion.to(device)  # move bin_centers and all buffers to GPU

    # 6. Optimizer + LR schedule
    tcfg = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    total_steps = len(train_loader) * tcfg["num_epochs"] // tcfg["gradient_accumulation_steps"]
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 7. WandB
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["experiment_name"],
                config=cfg,
            )
            logger.info("WandB initialized")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Disabling.")
            use_wandb = False

    # 8. Resume checkpoint
    global_step = 0
    start_epoch = 0
    if resume_path:
        # resume_path can be a directory ("checkpoint-last" / "checkpoint-best")
        # or the keyword "last" / "best" to auto-resolve inside out_dir
        if resume_path in ("last", "best"):
            ckpt = ckpt_manager.load(resume_path, device)
        else:
            ckpt_file = Path(resume_path) / "checkpoint.pt"
            ckpt = torch.load(ckpt_file, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        if use_ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        # Restore best-metric baseline so "best" tracking continues correctly
        if "best_value" in ckpt and ckpt["best_value"] is not None:
            ckpt_manager.best_value = ckpt["best_value"]
        logger.info(
            f"Resumed from step {global_step}, epoch {start_epoch} "
            f"(best {ckpt_manager.metric}={ckpt_manager.best_value})"
        )

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    cfg_dropout_rate = difcfg["cfg_dropout_rate"]
    grad_accum_steps = tcfg["gradient_accumulation_steps"]
    max_grad_norm    = tcfg["gradient_clip_norm"]

    logger.info(f"Starting training: {tcfg['num_epochs']} epochs, "
                f"{len(train_loader)} steps/epoch")

    for epoch in range(start_epoch, tcfg["num_epochs"]):
        model.train()
        epoch_losses = {"total": 0.0, "l1": 0.0, "perceptual": 0.0,
                        "gradient": 0.0, "histogram": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['num_epochs']}")
        optimizer.zero_grad()

        for step_in_epoch, batch in enumerate(pbar):
            # ---- Move batch to device (all tensors) ----
            batch = to_device(batch, device)

            normal_y_norm = batch["normal_y_norm"].to(amp_dtype)   # (B,1,H,W) [-1,1]
            normal_y      = batch["normal_y"].to(amp_dtype)        # (B,1,H,W) [0,1]
            ref_hist      = batch["ref_hist"].to(amp_dtype)        # (B,256)
            labels        = batch["label"]                         # (B,) long, already on device

            # ---- CFG: randomly drop conditioning ----
            # Replace labels with null class for cfg_dropout_rate fraction
            if cfg_dropout_rate > 0:
                drop_mask = torch.rand(labels.shape[0], device=device) < cfg_dropout_rate
                labels = labels.clone()
                labels[drop_mask] = 2  # 2 = NULL class

            # ---- Sample noise and timestep ----
            noise     = torch.randn_like(normal_y_norm)
            timesteps = sample_timesteps(
                normal_y_norm.shape[0],
                difcfg["num_train_timesteps"],
                device,
            )

            # ---- Forward diffusion: add noise ----
            noisy = add_noise(scheduler, normal_y_norm, noise, timesteps)

            # ---- Model forward (AMP context) ----
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_noise = model(
                    x              = noisy,
                    timesteps      = timesteps,
                    exposure_label = labels,
                    condition_y    = normal_y_norm,
                )

                # Reconstruct x0 for pixel-space losses
                pred_x0_norm = predict_x0(
                    noisy, pred_noise, timesteps, alphas_cumprod, device
                )
                pred_x0  = denormalize_luminance(pred_x0_norm)   # [0, 1]

                # ---- Loss ----
                loss_dict = criterion(
                    pred_noise   = pred_noise,
                    target_noise = noise,
                    pred_x0      = pred_x0,
                    target_x0    = normal_y,        # normal frame as structural ref
                    ref_hist     = ref_hist,
                )
                loss = loss_dict["total"] / grad_accum_steps

            # ---- Backward ----
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step_in_epoch + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

                if use_ema:
                    ema.step(model)

                global_step += 1

                # ---- Logging ----
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v.item()

                if global_step % tcfg["log_every"] == 0:
                    log_dict = {
                        f"train/{k}": v.item() for k, v in loss_dict.items()
                    }
                    log_dict["train/lr"] = lr_scheduler.get_last_lr()[0]
                    log_dict["train/epoch"] = epoch

                    pbar.set_postfix({
                        "loss": f"{loss_dict['total'].item():.4f}",
                        "l1":   f"{loss_dict['l1'].item():.4f}",
                        "hist": f"{loss_dict['histogram'].item():.4f}",
                        "lr":   f"{lr_scheduler.get_last_lr()[0]:.6f}",
                    })

                    if use_wandb:
                        import wandb
                        wandb.log(log_dict, step=global_step)

                # ---- Sample visualization ----
                if global_step % tcfg["sample_every"] == 0:
                    try:
                        save_sample_grid(
                            model       = model,
                            scheduler   = scheduler,
                            val_batch   = val_batch_fixed,
                            device      = device,
                            save_path   = sample_dir / f"step_{global_step:06d}.png",
                            cfg_scale   = difcfg["cfg_scale"],
                            num_inference_steps = 20,
                            dtype       = amp_dtype,
                        )
                        empty_cache()
                    except Exception as e:
                        logger.warning(f"Sample generation failed: {e}")

                # ---- Checkpoint: save "last" (overwrites previous) ----
                if global_step % tcfg["save_every"] == 0:
                    ckpt_manager.save_last(
                        model=model, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, scaler=scaler,
                        ema=ema, global_step=global_step,
                        epoch=epoch, cfg=cfg,
                    )

        # ---- End of epoch validation ----
        model.eval()
        val_metrics_accum = {}
        n_val_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_batch       = to_device(val_batch, device)
                v_normal_y_norm = val_batch["normal_y_norm"].to(amp_dtype)
                v_label         = val_batch["label"]
                v_ref_hist      = val_batch["ref_hist"]

                with ema.average_parameters(model) if use_ema else _noop():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred_y = ddim_sample(
                            model           = model,
                            scheduler       = ddim_val,
                            shape           = v_normal_y_norm.shape,
                            exposure_labels = v_label,
                            condition_images= v_normal_y_norm,
                            device          = device,
                            cfg_scale       = difcfg["cfg_scale"],
                            dtype           = amp_dtype,
                        )

                pred_y_01   = denormalize_luminance(pred_y)
                normal_y_01 = val_batch["normal_y"].to(amp_dtype)

                batch_metrics = evaluate_batch(
                    pred_y   = pred_y_01,
                    normal_y = normal_y_01,
                    label    = v_label,
                    ref_hist = v_ref_hist.to(amp_dtype),
                )
                for k, v in batch_metrics.items():
                    val_metrics_accum[k] = val_metrics_accum.get(k, 0) + v
                n_val_batches += 1

                empty_cache()
                break  # 1 val batch per epoch is enough for metric tracking

        val_metrics = {k: v / max(n_val_batches, 1) for k, v in val_metrics_accum.items()}

        logger.info(
            f"Epoch {epoch+1} | "
            f"EVS: {val_metrics.get('exposure_visibility', 0):.3f} | "
            f"SSIM: {val_metrics.get('ssim_vs_normal', 0):.3f} | "
            f"Lum mean: {val_metrics.get('lum_mean', 0):.3f} | "
            f"Hist KL: {val_metrics.get('hist_kl_div', 0):.4f}"
        )

        # ---- Per-epoch sample image ----
        try:
            with ema.average_parameters(model) if use_ema else _noop():
                save_sample_grid(
                    model               = model,
                    scheduler           = scheduler,
                    val_batch           = val_batch_fixed,
                    device              = device,
                    save_path           = sample_dir / f"epoch_{epoch+1:04d}.png",
                    cfg_scale           = difcfg["cfg_scale"],
                    num_inference_steps = 20,
                    dtype               = amp_dtype,
                )
            logger.info(f"  Sample image → {sample_dir}/epoch_{epoch+1:04d}.png")
            empty_cache()
        except Exception as e:
            logger.warning(f"Per-epoch sample generation failed: {e}")

        # ---- Checkpoint: save "last" every epoch ----
        ckpt_manager.save_last(
            model=model, optimizer=optimizer,
            lr_scheduler=lr_scheduler, scaler=scaler,
            ema=ema, global_step=global_step,
            epoch=epoch, cfg=cfg,
        )

        # ---- Checkpoint: save "best" if metric improved ----
        improved = ckpt_manager.save_best_if_improved(
            model=model, optimizer=optimizer,
            lr_scheduler=lr_scheduler, scaler=scaler,
            ema=ema, global_step=global_step,
            epoch=epoch, cfg=cfg,
            val_metrics=val_metrics,
        )
        if improved:
            logger.info(
                f"  ★ New best {ckpt_manager.metric}={ckpt_manager.best_value:.4f}"
            )

        if use_wandb:
            import wandb
            wandb_log = {f"val/{k}": v for k, v in val_metrics.items()}
            sample_path = sample_dir / f"epoch_{epoch+1:04d}.png"
            if sample_path.exists():
                wandb_log["val/sample"] = wandb.Image(str(sample_path))
            wandb.log(wandb_log, step=global_step)

        log_gpu_memory(f"epoch {epoch+1} end")

    # ---- Final save ----
    final_dir = base_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        **({"ema": ema.state_dict()} if use_ema else {}),
        "config": cfg,
        "global_step": global_step,
    }, final_dir / "model_final.pt")
    logger.info(f"Training complete. Final model → {final_dir}")

    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# No-op context manager (for when EMA is disabled)
# ---------------------------------------------------------------------------

@contextmanager
def _noop():
    yield


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Illumination Diffusion Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint. Use 'last', 'best', or a directory path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {cfg['experiment_name']}")

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
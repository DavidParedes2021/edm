"""
training/trainer.py
Main training loop for the illumination diffusion model.

Features:
  - Mixed-precision (fp16) via torch.cuda.amp  [avoids diffusers accelerate conflicts]
  - Gradient accumulation (effective large batch on single GPU)
  - EMA weight averaging
  - Validation loss + sample generation every N epochs
  - W&B logging (optional)
  - Device-safe: all tensors moved to device before any operation
  - Memory-safe: explicit del + empty_cache between heavy ops

Compatible with:
  diffusers==0.14.0, transformers==4.27.4, accelerate==0.18.0, wandb==0.14.2
"""

import os
import math
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

# ── Local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet      import IlluminationUNet, EMA, build_model
from models.diffusion  import DDPMScheduler, DDIMSampler, build_scheduler, build_sampler
from models.losses     import HybridDiffusionLoss
from data.dataset      import build_dataloaders
from training.config_utils import load_config, set_seed, get_device, make_output_dir

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler_lr,
    scaler: GradScaler,
    ema: Optional[EMA],
    best_val_loss: float,
):
    state = {
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "best_val_loss":  best_val_loss,
    }
    if scheduler_lr is not None:
        state["lr_scheduler"] = scheduler_lr.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    ema: Optional[EMA],
    scheduler_lr,
    device: torch.device,
):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    if scheduler_lr is not None and "lr_scheduler" in state:
        scheduler_lr.load_state_dict(state["lr_scheduler"])
    epoch          = state["epoch"]
    best_val_loss  = state.get("best_val_loss", float("inf"))
    logger.info(f"Resumed from epoch {epoch}: {path}")
    return epoch, best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [3,H,W] tensor in [-1,1] to PIL Image."""
    arr = ((t.float().clamp(-1, 1) + 1.0) * 0.5 * 255).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    sampler: DDIMSampler,
    batch: dict,
    device: torch.device,
    n_samples: int = 4,
) -> dict:
    """
    Generate a small grid of (normal | generated | GT artifact) for logging.
    Returns dict of PIL images keyed by label.
    """
    model.eval()
    n   = min(n_samples, batch["normal"].shape[0])
    cond      = batch["normal"][:n].to(device)
    exposure  = batch["exposure"][:n].to(device)
    gt        = batch["artifact"][:n].to(device)
    B, C, H, W = cond.shape

    generated = sampler.sample(
        model    = model,
        shape    = (B, C, H, W),
        cond     = cond,
        exposure = exposure,
        device   = device,
    )

    images = {}
    for i in range(n):
        images[f"normal_{i}"]    = tensor_to_pil(cond[i])
        images[f"generated_{i}"] = tensor_to_pil(generated[i])
        images[f"gt_{i}"]        = tensor_to_pil(gt[i])

    # Free GPU memory
    del generated, cond, exposure, gt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    return images


def save_sample_grid(images: dict, path: str, n: int = 4):
    """Save a horizontal strip: [normal | generated | GT] × n."""
    keys = [f"normal_{i}" for i in range(n) if f"normal_{i}" in images]
    if not keys:
        return
    H, W = images[keys[0]].size[1], images[keys[0]].size[0]
    grid = Image.new("RGB", (W * 3 * len(keys), H))
    for i, k in enumerate(keys):
        norm = images.get(f"normal_{i}")
        gen  = images.get(f"generated_{i}")
        gt_  = images.get(f"gt_{i}")
        if norm: grid.paste(norm, (i * W * 3,       0))
        if gen:  grid.paste(gen,  (i * W * 3 + W,   0))
        if gt_:  grid.paste(gt_,  (i * W * 3 + 2*W, 0))
    grid.save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loss_fn: HybridDiffusionLoss,
    batch: dict,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler,
    optimizer: optim.Optimizer,
    grad_accum_steps: int,
    step_in_accum: int,
    max_grad_norm: float,
    ema: Optional[EMA],
) -> dict:
    """
    One forward-backward pass (may be inside a gradient accumulation loop).
    Returns dict of loss values.
    """
    # ── Move batch to device ──────────────────────────────────────────
    normal   = batch["normal"].to(device, non_blocking=True)      # [B,3,H,W]
    artifact = batch["artifact"].to(device, non_blocking=True)    # [B,3,H,W]
    exposure = batch["exposure"].to(device, non_blocking=True)    # [B]

    B = normal.shape[0]

    # ── Sample random timesteps ───────────────────────────────────────
    t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)

    # ── Add noise to ground-truth artifact (x_0 = artifact) ──────────
    noise  = torch.randn_like(artifact)
    xt     = scheduler.add_noise(artifact, noise, t)              # x_t

    # ── Target for loss ───────────────────────────────────────────────
    target = scheduler.get_target(artifact, noise, t)             # v or ε

    # ── Forward pass ─────────────────────────────────────────────────
    is_last_accum = (step_in_accum + 1) == grad_accum_steps

    with autocast(enabled=use_amp):
        model_out = model(xt, t, normal, exposure)               # [B,3,H,W]

        # Predict x0 for perceptual loss
        x0_pred = scheduler.predict_x0(xt, model_out, t).detach()  # detach for perc loss
        loss, log = loss_fn(model_out, target, x0_pred, artifact)
        loss = loss / grad_accum_steps                           # scale for accumulation

    # ── Backward pass ─────────────────────────────────────────────────
    scaler.scale(loss).backward()

    if is_last_accum:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()

    return log


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loss_fn: HybridDiffusionLoss,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 50,
) -> float:
    """Compute mean val loss (MSE only for speed; no full sampling)."""
    model.eval()
    total_loss = 0.0
    count      = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        normal   = batch["normal"].to(device, non_blocking=True)
        artifact = batch["artifact"].to(device, non_blocking=True)
        exposure = batch["exposure"].to(device, non_blocking=True)
        B        = normal.shape[0]

        t      = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
        noise  = torch.randn_like(artifact)
        xt     = scheduler.add_noise(artifact, noise, t)
        target = scheduler.get_target(artifact, noise, t)

        with autocast(enabled=use_amp):
            model_out = model(xt, t, normal, exposure)
            x0_pred   = scheduler.predict_x0(xt, model_out, t)
            loss, _   = loss_fn(model_out, target, x0_pred, artifact)

        total_loss += loss.item() * B
        count      += B

    model.train()
    return total_loss / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main Trainer class
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.device  = get_device()
        set_seed(cfg.get("seed", 42))
        self.out_dir = make_output_dir(cfg)

        logging.basicConfig(
            level   = logging.INFO,
            format  = "%(asctime)s [%(levelname)s] %(message)s",
            handlers= [
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.out_dir, "logs", "train.log")),
            ],
        )

        logger.info(f"Device: {self.device}")
        logger.info(f"Output dir: {self.out_dir}")

        # ── W&B ───────────────────────────────────────────────────────
        self.use_wandb = cfg["training"].get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project = cfg.get("experiment_name", "illumination_diffusion"),
                    config  = cfg,
                )
                self.wandb = wandb
            except Exception as e:
                logger.warning(f"W&B init failed: {e}. Disabling W&B.")
                self.use_wandb = False

        # ── Build components ─────────────────────────────────────────
        self.model     = build_model(cfg).to(self.device)
        self.scheduler = build_scheduler(cfg)
        self.sampler   = build_sampler(cfg, self.scheduler)
        self.loss_fn   = HybridDiffusionLoss(
            mse_weight        = cfg["training"].get("mse_weight", 1.0),
            perceptual_weight = cfg["training"].get("perceptual_weight", 0.0),
        )

        model_cfg  = cfg["model"]
        self.ema: Optional[EMA] = None
        if model_cfg.get("use_ema", False):
            self.ema = EMA(self.model, decay=model_cfg.get("ema_decay", 0.9999))

        # ── Optimizer & LR scheduler ─────────────────────────────────
        train_cfg = cfg["training"]
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr           = train_cfg["learning_rate"],
            betas        = (0.9, 0.999),
            weight_decay = 1e-4,
        )
        total_steps = (
            len(build_dataloaders(cfg, "train")) // train_cfg["gradient_accumulation_steps"]
            * train_cfg["num_epochs"]
        )
        warmup_steps = train_cfg.get("lr_warmup_steps", 500)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr         = train_cfg["learning_rate"],
            total_steps    = total_steps,
            pct_start      = min(warmup_steps / max(total_steps, 1), 0.3),
            anneal_strategy= "cos",
        )

        # ── AMP ───────────────────────────────────────────────────────
        self.use_amp = (
            train_cfg.get("mixed_precision", "no") == "fp16"
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        logger.info(f"AMP enabled: {self.use_amp}")

        # ── Params ───────────────────────────────────────────────────
        self.grad_accum  = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_epochs  = train_cfg["num_epochs"]
        self.save_every  = train_cfg.get("save_every_n_epochs", 10)
        self.log_every   = train_cfg.get("log_every_n_steps", 50)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

        self.global_step  = 0
        self.best_val_loss = float("inf")
        self.start_epoch   = 0

        # Check for existing checkpoint to resume
        last_ckpt = os.path.join(self.out_dir, "checkpoints", "last.pt")
        if os.path.exists(last_ckpt):
            self.start_epoch, self.best_val_loss = load_checkpoint(
                last_ckpt, self.model, self.optimizer,
                self.scaler, self.ema, self.lr_scheduler, self.device,
            )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────

    def fit(self):
        train_loader = build_dataloaders(self.cfg, "train")
        val_loader   = build_dataloaders(self.cfg, "validation")

        # Get a fixed val batch for visual samples
        try:
            sample_batch = next(iter(val_loader))
        except StopIteration:
            sample_batch = next(iter(train_loader))

        logger.info(
            f"Training: {len(train_loader)} batches/epoch, "
            f"{self.max_epochs} epochs, "
            f"grad_accum={self.grad_accum}"
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_loss   = 0.0
            step_in_accum = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            for batch in pbar:
                log = train_step(
                    model           = self.model,
                    scheduler       = self.scheduler,
                    loss_fn         = self.loss_fn,
                    batch           = batch,
                    device          = self.device,
                    use_amp         = self.use_amp,
                    scaler          = self.scaler,
                    optimizer       = self.optimizer,
                    grad_accum_steps= self.grad_accum,
                    step_in_accum   = step_in_accum,
                    max_grad_norm   = self.max_grad_norm,
                    ema             = self.ema,
                )

                step_in_accum = (step_in_accum + 1) % self.grad_accum
                if step_in_accum == 0:
                    self.lr_scheduler.step()
                    self.global_step += 1

                epoch_loss += log["total"]
                pbar.set_postfix({
                    "loss": f"{log['total']:.4f}",
                    "lr":   f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

                # ── Step logging ──────────────────────────────────────
                if self.global_step % self.log_every == 0 and step_in_accum == 0:
                    if self.use_wandb:
                        self.wandb.log({
                            "train/loss": log["total"],
                            "train/mse":  log.get("mse", 0),
                            "train/perceptual": log.get("perceptual", 0),
                            "train/lr":   self.optimizer.param_groups[0]["lr"],
                            "step": self.global_step,
                        })

            # ── End of epoch ──────────────────────────────────────────
            mean_train = epoch_loss / len(train_loader)
            val_loss   = validate(
                self.model, self.scheduler, self.loss_fn,
                val_loader, self.device, self.use_amp,
            )
            logger.info(
                f"Epoch {epoch+1}: train_loss={mean_train:.4f}, val_loss={val_loss:.4f}"
            )

            if self.use_wandb:
                self.wandb.log({"val/loss": val_loss, "epoch": epoch + 1})

            # ── Save checkpoints ───────────────────────────────────────
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(self.out_dir, "checkpoints", "best.pt"),
                    epoch + 1, self.model, self.optimizer, self.lr_scheduler,
                    self.scaler, self.ema, self.best_val_loss,
                )

            if (epoch + 1) % self.save_every == 0:
                save_checkpoint(
                    os.path.join(self.out_dir, "checkpoints", f"epoch_{epoch+1:04d}.pt"),
                    epoch + 1, self.model, self.optimizer, self.lr_scheduler,
                    self.scaler, self.ema, self.best_val_loss,
                )

            save_checkpoint(
                os.path.join(self.out_dir, "checkpoints", "last.pt"),
                epoch + 1, self.model, self.optimizer, self.lr_scheduler,
                self.scaler, self.ema, self.best_val_loss,
            )

            # ── Visual samples ─────────────────────────────────────────
            if (epoch + 1) % self.save_every == 0:
                # Use EMA weights if available
                if self.ema is not None:
                    self.ema.apply_shadow()

                images = generate_samples(
                    self.model, self.scheduler, self.sampler,
                    sample_batch, self.device,
                )
                grid_path = os.path.join(
                    self.out_dir, "samples", f"epoch_{epoch+1:04d}.png"
                )
                save_sample_grid(images, grid_path)
                logger.info(f"Sample grid saved: {grid_path}")

                if self.ema is not None:
                    self.ema.restore()

                if self.use_wandb:
                    import wandb as _wandb
                    self.wandb.log({
                        "samples": _wandb.Image(grid_path),
                        "epoch": epoch + 1,
                    })

            # ── Memory cleanup ─────────────────────────────────────────
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Training complete.")
        if self.use_wandb:
            self.wandb.finish()

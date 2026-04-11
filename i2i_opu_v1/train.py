#!/usr/bin/env python3
"""
train.py — Train luminance-only conditional DDPM for exposure augmentation.

Architecture:
    - UNet2DModel (diffusers) operating on the L channel (1-channel in/out)
    - Class conditioning: 0=overexposed, 1=underexposed, 2=unconditional
    - Classifier-free guidance via random label dropout during training

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume output/checkpoints/latest.pt
"""

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from tqdm import tqdm

import yaml

from dataset import ExposureDataset, NormalImageDataset, lab_to_rgb_numpy
from losses import SobelEdgeLoss, VGGPerceptualLoss
from PIL import Image

# ---- config loading ------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---- EMA ------------------------------------------------------------------ #

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


# ---- build model ---------------------------------------------------------- #

def build_model(cfg: dict, device: torch.device) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=cfg["image"]["size"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        layers_per_block=cfg["model"]["layers_per_block"],
        down_block_types=tuple(cfg["model"]["down_block_types"]),
        up_block_types=tuple(cfg["model"]["up_block_types"]),
        num_class_embeds=cfg["model"]["num_class_embeds"],
        attention_head_dim=cfg["model"]["attention_head_dim"],
    )
    model = model.to(device)
    return model


# ---- sampling for visualisation ------------------------------------------- #

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    scheduler_cfg: dict,
    normal_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    epoch: int,
    output_dir: str,
    num_samples: int = 4,
):
    """SDEdit: add noise to normal L channel, denoise with domain conditioning."""
    model.eval()

    ddim = DDIMScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
        beta_schedule=scheduler_cfg["beta_schedule"],
        prediction_type=scheduler_cfg["prediction_type"],
    )
    ddim.set_timesteps(cfg["inference"]["num_inference_steps"], device=device)

    noise_strength = cfg["inference"]["noise_strength"]
    guidance_scale = cfg["inference"]["guidance_scale"]

    # how many forward-noising steps to apply
    total_steps = len(ddim.timesteps)
    start_step = int(total_steps * (1 - noise_strength))
    start_step = max(0, min(start_step, total_steps - 1))

    samples_dir = Path(output_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for L_tensor, AB_tensor, paths in normal_loader:
        if count >= num_samples:
            break
        L_tensor = L_tensor.to(device)  # (B, 1, H, W)
        AB_np = AB_tensor.numpy()       # (B, H, W, 2)

        B = L_tensor.shape[0]

        for domain_label, domain_name in [(0, "overexposed"), (1, "underexposed")]:
            # add noise at the start_step level
            noise = torch.randn_like(L_tensor)
            t_start = ddim.timesteps[start_step]
            noisy_L = ddim.add_noise(L_tensor, noise, torch.tensor([t_start], device=device))

            x = noisy_L
            cond_labels = torch.full((B,), domain_label, dtype=torch.long, device=device)
            uncond_labels = torch.full((B,), 2, dtype=torch.long, device=device)

            for t in ddim.timesteps[start_step:]:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)

                # classifier-free guidance
                with autocast(enabled=cfg["training"]["mixed_precision"]):
                    pred_cond = model(x, t_batch, class_labels=cond_labels).sample
                    pred_uncond = model(x, t_batch, class_labels=uncond_labels).sample

                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                x = ddim.step(pred, t, x).prev_sample

            # denormalise L: [-1,1] → [0,100]
            L_out = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                if count >= num_samples:
                    break
                lab = np.stack([L_out[b], AB_np[b, :, :, 0], AB_np[b, :, :, 1]], axis=-1)
                rgb = lab_to_rgb_numpy(lab)
                fname = f"epoch{epoch:04d}_{Path(paths[b]).stem}_{domain_name}.png"
                Image.fromarray(rgb).save(str(samples_dir / fname))
                count += 1

    model.train()


# ---- training ------------------------------------------------------------- #

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")
    if device.type == "cuda":
        print(f"        GPU = {torch.cuda.get_device_name(0)}, "
              f"VRAM = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # directories
    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    samp_dir = Path(cfg["output"]["samples_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)

    # wandb (optional)
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            config=cfg,
        )

    # dataset & loader
    dataset = ExposureDataset(
        overexposed_dir=cfg["data"]["overexposed_dir"],
        underexposed_dir=cfg["data"]["underexposed_dir"],
        image_size=cfg["image"]["size"],
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # normal images for periodic sample generation
    normal_ds = NormalImageDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
    )
    normal_loader = DataLoader(normal_ds, batch_size=1, shuffle=True, num_workers=0)

    # model
    model = build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] parameters: {param_count / 1e6:.2f} M")

    # noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )

    # optimiser
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # EMA
    ema = EMAModel(model, decay=cfg["training"]["ema_decay"])

    # losses
    edge_loss_fn = SobelEdgeLoss().to(device)
    perceptual_loss_fn = None
    if cfg["losses"]["perceptual_weight"] > 0:
        perceptual_loss_fn = VGGPerceptualLoss(device)

    # resume
    start_epoch = 0
    best_loss = float("inf")

    # training loop
    cfg_drop = cfg["training"]["cfg_dropout_prob"]
    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["grad_accum_steps"]

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (L_batch, labels) in enumerate(pbar):
            L_batch = L_batch.to(device)   # (B, 1, H, W)
            labels = labels.to(device)     # (B,)

            # classifier-free guidance dropout
            mask = torch.rand(labels.shape[0], device=device) < cfg_drop
            labels = labels.clone()
            labels[mask] = 2  # unconditional label

            # sample noise
            noise = torch.randn_like(L_batch)
            B = L_batch.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()

            noisy = noise_scheduler.add_noise(L_batch, noise, timesteps)

            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred = model(noisy, timesteps, class_labels=labels).sample

                # main MSE loss
                loss = F.mse_loss(pred, noise) * cfg["losses"]["mse_weight"]

                # edge loss for sharpness (compare denoised estimate vs clean)
                if cfg["losses"]["edge_weight"] > 0:
                    # approximate x0 from noise prediction for edge comparison
                    alpha_prod = noise_scheduler.alphas_cumprod.to(device)[timesteps]
                    alpha_prod = alpha_prod.view(-1, 1, 1, 1)
                    x0_hat = (noisy - torch.sqrt(1 - alpha_prod) * pred) / torch.sqrt(alpha_prod).clamp(min=1e-8)
                    loss = loss + cfg["losses"]["edge_weight"] * edge_loss_fn(x0_hat, L_batch)

                # perceptual loss
                if perceptual_loss_fn is not None and cfg["losses"]["perceptual_weight"] > 0:
                    loss = loss + cfg["losses"]["perceptual_weight"] * perceptual_loss_fn(x0_hat, L_batch)

                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
                ema.update(model)

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[Epoch {epoch+1}] avg loss = {avg_loss:.5f}")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        # save latest checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
            "config": cfg,
        }
        torch.save(ckpt, str(ckpt_dir / "latest.pt"))

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, str(ckpt_dir / "best.pt"))
            print(f"  → new best model saved (loss={best_loss:.5f})")

        # periodic sample generation
        if (epoch + 1) % cfg["training"]["save_every_n_epochs"] == 0:
            # swap in EMA weights for generation
            orig_sd = copy.deepcopy(model.state_dict())
            ema.apply(model)
            generate_samples(
                model, cfg["diffusion"], normal_loader, device, cfg,
                epoch + 1, str(samp_dir), num_samples=4,
            )
            model.load_state_dict(orig_sd)
            print(f"  → samples saved to {samp_dir}")

    print("[Train] done.")
    if use_wandb:
        import wandb
        wandb.finish()


# ---- CLI ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        # update cfg from checkpoint if desired
        # cfg = ckpt["config"]  # uncomment to force original config

    train(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
diffusion_train.py — Train conditional DDPM on paired L-channel data.

Architecture (Palette-style):
    The UNet receives 2 input channels: concat(noisy_target_L, source_L).
    It predicts the noise added to target_L, conditioned on source_L and a
    domain class label (0=overexposed, 1=underexposed).

    This is a standard conditional diffusion model with paired supervision.
    The source L channel acts as a pixel-level condition — the model sees
    exactly which structures exist and learns only the luminance *shift*.

Why this preserves texture:
    - The model operates on L channel only (AB untouched by construction).
    - The source L is concatenated as input, so the model has direct access
      to the original structure and only needs to learn the residual shift.
    - Paired training with L1 + edge loss penalises any structural deviation.
    - At inference, the predicted L is blended at full resolution using the
      same texture-decomposition trick from exposure_augment.py.

Usage:
    # Step 1: generate paired data
    python generate_pairs.py --normal_dir ./data/normal --output_dir ./data/pairs

    # Step 2: train
    python diffusion_train.py --config diffusion_config.yaml

    # Step 3: generate
    python diffusion_inference.py --config diffusion_config.yaml \
        --checkpoint ./output_diffusion/checkpoints/best.pt
"""

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from PIL import Image
import yaml

from diffusion_dataset import PairedLuminanceDataset, NormalInferenceDataset
from exposure_augment import lab_to_rgb, rgb_to_lab
from losses import SobelEdgeLoss


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class EMAModel:
    """Exponential moving average of model parameters."""

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


def build_model(cfg: dict, device: torch.device) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=cfg["image"]["size"],
        in_channels=cfg["model"]["in_channels"],   # 2: noisy_target + source
        out_channels=cfg["model"]["out_channels"],  # 1: predicted noise
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        layers_per_block=cfg["model"]["layers_per_block"],
        down_block_types=tuple(cfg["model"]["down_block_types"]),
        up_block_types=tuple(cfg["model"]["up_block_types"]),
        num_class_embeds=cfg["model"]["num_class_embeds"],
        attention_head_dim=cfg["model"]["attention_head_dim"],
    )
    model = model.to(device)
    try:
        model.enable_gradient_checkpointing()
    except (ValueError, AttributeError):
        print("[Model] gradient checkpointing not supported — skipping")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Sample generation during training (for visual inspection)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    scheduler_cfg: dict,
    inference_cfg: dict,
    normal_loader: DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str,
    image_size: int,
    num_samples: int = 4,
):
    """Full DDIM denoising conditioned on source L, then recombine with AB."""
    model.eval()

    ddim = DDIMScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
        beta_schedule=scheduler_cfg["beta_schedule"],
        prediction_type=scheduler_cfg["prediction_type"],
    )
    ddim.set_timesteps(inference_cfg["num_inference_steps"], device=device)

    samples_dir = Path(output_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_L, AB, L_orig, paths, orig_hw in normal_loader:
        if count >= num_samples:
            break

        source_L = source_L.to(device)  # (1, 1, H, W)
        B = source_L.shape[0]

        for domain_label, domain_name in [(0, "overexposed"), (1, "underexposed")]:
            # start from pure noise for the target channel
            x = torch.randn_like(source_L)  # (B, 1, H, W)
            cond_labels = torch.full((B,), domain_label, dtype=torch.long, device=device)

            for t in ddim.timesteps:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)
                # concat source as conditioning channel
                model_input = torch.cat([x, source_L], dim=1)  # (B, 2, H, W)
                pred_noise = model(model_input, t_batch, class_labels=cond_labels).sample
                x = ddim.step(pred_noise, t, x).prev_sample

            # x is the predicted target L at model resolution, in [-1, 1]
            L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                if count >= num_samples:
                    break

                H_orig, W_orig = int(orig_hw[0][b]), int(orig_hw[1][b])
                ab = AB[b].numpy()  # (H_orig, W_orig, 2)

                # upsample predicted L to original resolution
                L_pred_full = np.array(
                    Image.fromarray(L_pred_small[b].astype(np.float32), mode="F").resize(
                        (W_orig, H_orig), Image.LANCZOS
                    ),
                    dtype=np.float32,
                )

                # texture preservation: extract high-pass from original L,
                # extract low-pass from predicted L, recombine
                from scipy.ndimage import gaussian_filter
                sigma = 3.0 * max(H_orig, W_orig) / 512.0
                L_orig_np = L_orig[b].numpy()
                L_high = L_orig_np - gaussian_filter(L_orig_np, sigma=sigma)
                L_low_pred = gaussian_filter(L_pred_full, sigma=sigma)
                L_final = np.clip(L_low_pred + L_high, 0.0, 100.0).astype(np.float32)

                lab = np.stack([L_final, ab[..., 0], ab[..., 1]], axis=-1)
                rgb = lab_to_rgb(lab)

                fname = f"epoch{epoch:04d}_{Path(paths[b]).stem}_{domain_name}.png"
                Image.fromarray(rgb).save(str(samples_dir / fname))
                count += 1

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")
    if device.type == "cuda":
        print(f"        GPU = {torch.cuda.get_device_name(0)}, "
              f"VRAM = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    samp_dir = Path(cfg["output"]["samples_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        import wandb
        wandb.init(project=cfg["logging"]["wandb_project"],
                   entity=cfg["logging"]["wandb_entity"], config=cfg)

    # dataset
    train_ds = PairedLuminanceDataset(
        pairs_dir=cfg["data"]["pairs_dir"],
        image_size=cfg["image"]["size"],
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # normal images for periodic visual inspection
    normal_ds = NormalInferenceDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
    )
    normal_loader = DataLoader(normal_ds, batch_size=1, shuffle=True, num_workers=0)

    # model
    model = build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {param_count / 1e6:.2f} M parameters")

    # scheduler
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

    # auxiliary losses
    edge_loss_fn = SobelEdgeLoss().to(device)

    # resume
    start_epoch = 0
    best_loss = float("inf")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["best_loss"]
        print(f"[Resume] from epoch {start_epoch}, best_loss={best_loss:.5f}")

    # training
    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    mse_w = cfg["losses"]["mse_weight"]
    edge_w = cfg["losses"]["edge_weight"]
    l1_w = cfg["losses"]["l1_weight"]

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (source_L, target_L, labels) in enumerate(pbar):
            source_L = source_L.to(device)   # (B, 1, H, W) — condition
            target_L = target_L.to(device)   # (B, 1, H, W) — what we learn to generate
            labels = labels.to(device)       # (B,) — domain label

            # add noise to the *target* L channel
            noise = torch.randn_like(target_L)
            B = target_L.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()
            noisy_target = noise_scheduler.add_noise(target_L, noise, timesteps)

            # model input: concat noisy target + clean source
            model_input = torch.cat([noisy_target, source_L], dim=1)  # (B, 2, H, W)

            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred_noise = model(model_input, timesteps, class_labels=labels).sample

                # primary loss: MSE on predicted noise
                loss = mse_w * F.mse_loss(pred_noise, noise)

                # auxiliary losses on the x0-estimate for sharpness
                alpha_prod = noise_scheduler.alphas_cumprod.to(device)[timesteps]
                alpha_prod = alpha_prod.view(-1, 1, 1, 1)
                sqrt_alpha = torch.sqrt(alpha_prod).clamp(min=1e-8)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_prod)
                x0_hat = (noisy_target - sqrt_one_minus * pred_noise) / sqrt_alpha

                # L1 loss: direct pixel supervision against paired target
                if l1_w > 0:
                    loss = loss + l1_w * F.l1_loss(x0_hat, target_L)

                # edge loss: preserve structural edges
                if edge_w > 0:
                    loss = loss + edge_w * edge_loss_fn(x0_hat, target_L)

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

        # save latest
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
            ckpt["best_loss"] = best_loss
            torch.save(ckpt, str(ckpt_dir / "best.pt"))
            print(f"  → best model saved (loss={best_loss:.5f})")

        # periodic samples
        if (epoch + 1) % cfg["training"]["save_every_n_epochs"] == 0:
            orig_sd = copy.deepcopy(model.state_dict())
            ema.apply(model)
            generate_samples(
                model, cfg["diffusion"], cfg["inference"],
                normal_loader, device, epoch + 1, str(samp_dir),
                cfg["image"]["size"], num_samples=4,
            )
            model.load_state_dict(orig_sd)
            print(f"  → samples saved to {samp_dir}")

    print("[Train] complete.")
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
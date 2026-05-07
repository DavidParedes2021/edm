"""Training script for the conditional I2I exposure diffusion model.

Usage:
    python -m i2i_diffusion.train --config i2i_diffusion/config.yaml
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset import PairDataset
from scheduler import DDPMScheduler, ddim_sample
from unet import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k in ssd.keys():
            if ssd[k].dtype.is_floating_point:
                ssd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                ssd[k].copy_(msd[k])


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    val_loader: DataLoader,
    device: torch.device,
    t_grid: list[int],
    max_batches: int = 0,
    seed: int = 0,
) -> float:
    """Compute mean ε-MSE on the val set with fixed (t, noise) for stability.

    For each batch and each t in t_grid we re-seed a CUDA generator with a
    deterministic value, so val_loss is reproducible across calls and
    sensitive to model improvements rather than RNG variance.
    """
    model.eval()
    total = 0.0
    count = 0
    for b_idx, batch in enumerate(val_loader):
        if max_batches > 0 and b_idx >= max_batches:
            break
        cond_L = batch["cond_L"].to(device)
        depth = batch["depth"].to(device)
        target_L = batch["target_L"].to(device)
        y = batch["mode"].to(device)
        B = target_L.shape[0]
        for ti, t_val in enumerate(t_grid):
            g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + b_idx * 131 + ti)
            t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
            noise = torch.randn(target_L.shape, device=device, generator=g)
            a = scheduler.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
            sm = scheduler.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
            x_t = a * target_L + sm * noise
            x_in = torch.cat([x_t, cond_L, depth], dim=1)
            eps_pred = model(x_in, t, y)
            mse = F.mse_loss(eps_pred, noise, reduction="mean").item()
            total += mse * B
            count += B
    model.train()
    return total / max(count, 1)


def save_preview(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch: Dict[str, torch.Tensor],
    out_path: Path,
    device: torch.device,
    num_steps: int = 50,
) -> None:
    """Sample for a single fixed batch and dump a side-by-side preview PNG."""
    model.eval()
    cond_L = batch["cond_L"].to(device)
    depth = batch["depth"].to(device)
    target_L = batch["target_L"].to(device)
    y = batch["mode"].to(device)

    cond = torch.cat([cond_L, depth], dim=1)
    pred = ddim_sample(
        model,
        scheduler,
        cond=cond,
        y=y,
        shape=target_L.shape,
        device=device,
        num_steps=num_steps,
        guidance_scale=1.0,
    )
    pred = torch.clamp(pred, -1.0, 1.0)
    model.train()

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        # (B, 1, H, W) in [-1, 1] → (B*H, W) uint8
        a = ((t.detach().cpu().float() + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        return a[:, 0]  # (B, H, W)

    cond_u = to_uint8(cond_L)
    target_u = to_uint8(target_L)
    pred_u = to_uint8(pred)
    rows = [np.concatenate([cond_u[i], target_u[i], pred_u[i]], axis=1) for i in range(cond_u.shape[0])]
    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="i2i_diffusion/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i2i] device={device}")

    # ── Data ──────────────────────────────────────────────────────────────
    val_fraction = float(cfg["data"].get("val_fraction", 0.0))
    split_seed = int(cfg["data"].get("split_seed", 1234))
    use_val = val_fraction > 0.0

    train_dataset = PairDataset(
        pairs_root=cfg["data"]["pairs_root"],
        resolution=cfg["data"]["resolution"],
        flip_prob=cfg["data"]["flip_prob"],
        split="train" if use_val else "all",
        val_fraction=val_fraction,
        split_seed=split_seed,
        augment=True,
    )
    val_dataset = (
        PairDataset(
            pairs_root=cfg["data"]["pairs_root"],
            resolution=cfg["data"]["resolution"],
            flip_prob=0.0,
            split="val",
            val_fraction=val_fraction,
            split_seed=split_seed,
            augment=False,
        )
        if use_val
        else None
    )
    n_val = len(val_dataset) if val_dataset is not None else 0
    print(f"[i2i] train items: {len(train_dataset)}  |  val items: {n_val}")
    if use_val and n_val == 0:
        print("[i2i] WARNING: val_fraction>0 but split produced 0 val items — "
              "best.pt will fall back to training-loss EMA.")
        val_dataset = None
        use_val = False

    loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    data_iter = cycle(loader)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg["data"].get("val_batch_size", cfg["training"]["batch_size"])),
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=cfg["data"]["num_workers"] > 0,
        )

    # fixed preview batch (taken from train set)
    preview_batch = next(iter(loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=3,                                  # noisy_L + cond_L + depth
        out_channels=1,
        base_channels=cfg["model"]["base_channels"],
        channel_mult=tuple(cfg["model"]["channel_mult"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        input_resolution=cfg["data"]["resolution"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[i2i] model params: {n_params/1e6:.2f} M")

    ema = EMA(model, decay=cfg["training"]["ema_decay"])

    # ── Diffusion ─────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        schedule=cfg["diffusion"]["schedule"],
    ).to(device)

    # ── Optim ─────────────────────────────────────────────────────────────
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["training"]["weight_decay"],
    )
    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_loss = float("inf")
    resume = cfg["training"].get("resume")
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        optim.load_state_dict(ck["optim"])
        start_step = int(ck.get("step", 0))
        best_loss = float(ck.get("best_loss", float("inf")))
        print(f"[i2i] resumed from {resume} @ step {start_step} (best={best_loss:.4f})")

    # ── Train loop ────────────────────────────────────────────────────────
    total_steps = cfg["training"]["total_steps"]
    log_every = cfg["training"]["log_every"]
    sample_every = cfg["training"]["sample_every"]
    ckpt_every = cfg["training"]["ckpt_every"]
    val_every = int(cfg["training"].get("val_every", ckpt_every))
    grad_clip = cfg["training"]["grad_clip"]
    drop_class_prob = float(cfg["data"].get("drop_class_prob", 0.0))
    val_t_grid = list(cfg["data"].get("val_t_grid", [50, 250, 500, 750, 950]))
    val_max_batches = int(cfg["data"].get("val_max_batches", 0))

    model.train()
    losses = []
    loss_ema: float | None = None
    loss_ema_alpha = 0.98
    t0 = time.time()
    for step in range(start_step, total_steps):
        batch = next(data_iter)
        cond_L = batch["cond_L"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        target_L = batch["target_L"].to(device, non_blocking=True)
        y = batch["mode"].to(device, non_blocking=True)

        # classifier-free dropout: replace some labels with the null token
        if drop_class_prob > 0:
            mask = torch.rand(y.shape, device=device) < drop_class_prob
            y = torch.where(mask, torch.full_like(y, model.num_classes), y)

        t = scheduler.sample_timesteps(target_L.shape[0], device)
        x_t, noise = scheduler.q_sample(target_L, t)
        x_in = torch.cat([x_t, cond_L, depth], dim=1)

        # update lr
        lr_now = cfg["training"]["lr"] * lr_lambda(step, cfg["training"]["warmup_steps"], total_steps)
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            eps_pred = model(x_in, t, y)
            loss = F.mse_loss(eps_pred, noise)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        ema.update(model)

        loss_val = loss.item()
        loss_ema = (
            loss_val if loss_ema is None
            else loss_ema_alpha * loss_ema + (1.0 - loss_ema_alpha) * loss_val
        )
        losses.append(loss_val)
        if (step + 1) % log_every == 0:
            mean_loss = sum(losses) / len(losses)
            losses.clear()
            elapsed = time.time() - t0
            it_s = log_every / max(elapsed, 1e-6)
            t0 = time.time()
            print(
                f"[i2i] step {step+1:>7d}/{total_steps} | loss {mean_loss:.4f} | "
                f"lr {lr_now:.2e} | {it_s:.2f} it/s"
            )

        if (step + 1) % sample_every == 0 or (step + 1) == total_steps:
            save_preview(
                ema.shadow,
                scheduler,
                preview_batch,
                out_path=out_dir / "samples" / f"step_{step+1:07d}.png",
                device=device,
            )

        # ── Validation + best.pt gating ───────────────────────────────────
        do_val = val_loader is not None and (
            (step + 1) % val_every == 0 or (step + 1) == total_steps
        )
        val_loss: float | None = None
        if do_val:
            val_loss = validate(
                ema.shadow,
                scheduler,
                val_loader,
                device,
                t_grid=val_t_grid,
                max_batches=val_max_batches,
            )

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            # rolling latest (full state) for crash recovery — single file, overwritten
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "optim": optim.state_dict(),
                    "config": cfg,
                    "best_loss": best_loss,
                },
                out_dir / "latest.pt",
            )
            # Pick the criterion: val loss when available, else smoothed train loss.
            if val_loss is not None:
                criterion = val_loss
                tag = "val"
            else:
                criterion = loss_ema if loss_ema is not None else float("inf")
                tag = "loss_ema"

            if criterion < best_loss:
                best_loss = criterion
                torch.save(
                    {
                        "step": step + 1,
                        "model": ema.shadow.state_dict(),
                        tag: best_loss,
                        "config": cfg,
                    },
                    out_dir / "best.pt",
                )
                print(
                    f"[i2i] new best @ step {step+1} ({tag}={best_loss:.4f}) -> best.pt"
                )
            else:
                cur = f"{criterion:.4f}" if criterion != float('inf') else "n/a"
                print(
                    f"[i2i] step {step+1} latest.pt "
                    f"({tag}={cur} | best={best_loss:.4f})"
                )

    print("[i2i] training complete.")


if __name__ == "__main__":
    main()

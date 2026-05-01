"""Train YCLDI: class-conditional luminance diffusion.

Usage
-----
    python train.py --config config.yaml
    python train.py --config config.yaml --resume

The training objective is the standard ε-prediction MSE on the Y channel,
with class-label conditioning and per-sample class dropout for CFG.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from diffusers import DDIMScheduler, DDPMScheduler

from checkpoint import CheckpointManager
from dataset import (
    CLASS_NORMAL,
    CLASS_NULL,
    CLASS_OVER,
    CLASS_UNDER,
    YCbCrEndoscopyDataset,
)
from ema import EMA
from model import build_unet
from utils import (
    derive_output_paths,
    humanize_param_count,
    load_config,
    save_sample_grid,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr_lambda(warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return fn


@torch.no_grad()
def quick_class_samples(
    unet,
    scheduler_inf: DDIMScheduler,
    device: torch.device,
    image_size: int,
    n_per_class: int = 2,
    cfg_scale: float = 4.0,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate samples from pure noise for each non-null class (over, under).

    Returns a tensor (n_classes * n_per_class, 3, H, W) in [0, 1] -- the Y
    channels visualized as grayscale RGB. This is just a training sanity check;
    the real artifact is produced by infer.py via SDEdit on a normal frame.
    """
    unet.eval()
    scheduler_inf.set_timesteps(num_steps)
    timesteps = scheduler_inf.timesteps.to(device)

    out_chunks = []
    for cls in (CLASS_OVER, CLASS_UNDER):
        x = torch.randn(n_per_class, 1, image_size, image_size, device=device)
        labels = torch.full((n_per_class,), cls, device=device, dtype=torch.long)
        null_labels = torch.full((n_per_class,), CLASS_NULL, device=device, dtype=torch.long)

        for t in timesteps:
            eps_cond = unet(x, t, class_labels=labels).sample
            eps_uncond = unet(x, t, class_labels=null_labels).sample
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            x = scheduler_inf.step(eps, t, x).prev_sample

        y01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        out_chunks.append(y01.repeat(1, 3, 1, 1))   # 1ch -> 3ch grayscale RGB

    unet.train()
    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output.root>/checkpoints/checkpoint-last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    paths = derive_output_paths(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] CUDA not available -- training on CPU will be very slow.")

    # -------- W&B (optional) ------------------------------------------------
    use_wandb = bool(cfg.get("wandb", {}).get("enabled", False))
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["wandb"].get("project", "ycldi"),
            entity=cfg["wandb"].get("entity"),
            name=cfg["project"].get("name"),
            config=cfg,
            dir=str(paths["logs"]),
        )

    # -------- Dataset / loaders --------------------------------------------
    dataset = YCbCrEndoscopyDataset(
        normal_dir=cfg["paths"]["normal_dir"],
        over_dir=cfg["paths"]["over_dir"],
        under_dir=cfg["paths"]["under_dir"],
        image_size=int(cfg["data"]["image_size"]),
        augment=bool(cfg["data"].get("augment", True)),
    )
    print(f"[data] total={len(dataset)} class_counts={dataset.class_counts()}")

    val_frac = float(cfg["data"].get("val_fraction", 0.05))
    n_val = max(2, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg["project"].get("seed", 42))),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 4))
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw,
        drop_last=True, pin_memory=pin, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=nw,
        drop_last=False, pin_memory=pin, persistent_workers=nw > 0,
    )

    # -------- Model ---------------------------------------------------------
    unet = build_unet(
        image_size=int(cfg["data"]["image_size"]),
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        layers_per_block=int(cfg["model"]["layers_per_block"]),
        attention_head_dim=int(cfg["model"]["attention_head_dim"]),
        norm_num_groups=int(cfg["model"]["norm_num_groups"]),
        resnet_time_scale_shift=str(cfg["model"]["resnet_time_scale_shift"]),
        num_class_embeds=4,                      # 3 real + null
    ).to(device)
    print(f"[model] params={humanize_param_count(unet)}")

    # Memory-efficient attention (best-effort)
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("[model] xformers attention enabled.")
    except Exception as e:                        # noqa: BLE001
        print(f"[model] xformers unavailable, using default attention. ({type(e).__name__})")

    # Gradient checkpointing for the 4 GB local smoke test
    if bool(cfg["train"].get("gradient_checkpointing", False)):
        try:
            unet.enable_gradient_checkpointing()
            print("[model] gradient checkpointing enabled.")
        except Exception as e:                    # noqa: BLE001
            print(f"[model] gradient checkpointing unavailable ({type(e).__name__}).")

    # -------- Schedulers ---------------------------------------------------
    scheduler_train = DDPMScheduler(
        num_train_timesteps=int(cfg["scheduler"]["num_train_timesteps"]),
        beta_start=float(cfg["scheduler"]["beta_start"]),
        beta_end=float(cfg["scheduler"]["beta_end"]),
        beta_schedule=str(cfg["scheduler"]["beta_schedule"]),
        prediction_type=str(cfg["scheduler"]["prediction_type"]),
    )
    # Move scheduler tensors to device once -- prevents the classic
    # "alphas_cumprod CPU/CUDA device mismatch" error inside add_noise().
    scheduler_train.alphas_cumprod = scheduler_train.alphas_cumprod.to(device)

    scheduler_inf = DDIMScheduler.from_config(scheduler_train.config)
    scheduler_inf.alphas_cumprod = scheduler_inf.alphas_cumprod.to(device)
    if hasattr(scheduler_inf, "final_alpha_cumprod"):
        scheduler_inf.final_alpha_cumprod = scheduler_inf.final_alpha_cumprod.to(device)

    # -------- Optim / EMA / AMP --------------------------------------------
    optim = AdamW(
        unet.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.999),
    )
    lr_sched = LambdaLR(optim, lr_lambda=get_lr_lambda(int(cfg["train"]["lr_warmup_steps"])))
    ema = EMA(unet, decay=float(cfg["train"]["ema_decay"]))

    use_amp = (
        str(cfg["train"].get("mixed_precision", "fp16")) == "fp16" and device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------- Checkpoints / resume -----------------------------------------
    ckpt_mgr = CheckpointManager(paths["ckpt"])
    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if args.resume:
        ck = ckpt_mgr.load_last(map_location="cpu")
        if ck is None:
            print("[resume] no checkpoint found; starting fresh.")
        else:
            unet.load_state_dict(ck["unet"])
            ema.load_state_dict(ck["ema"])
            optim.load_state_dict(ck["optim"])
            lr_sched.load_state_dict(ck["lr_sched"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck["epoch"]) + 1
            global_step = int(ck["global_step"])
            best_val = float(ck.get("best_val", best_val))
            print(f"[resume] resumed from epoch {start_epoch} (step {global_step}).")

    # -------- Training -----------------------------------------------------
    class_dropout_p = float(cfg["train"].get("class_dropout_prob", 0.1))
    log_every = int(cfg["train"].get("log_every", 50))
    sample_every = int(cfg["train"].get("sample_every", 1))
    ckpt_every = int(cfg["train"].get("ckpt_every", 1))
    grad_accum = max(1, int(cfg["train"].get("gradient_accumulation", 1)))
    num_epochs = int(cfg["train"]["num_epochs"])
    image_size = int(cfg["data"]["image_size"])

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0.0
        n_batches = 0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            y = batch["y"].to(device, non_blocking=True)            # (B, 1, H, W) in [-1, 1]
            labels = batch["label"].to(device, non_blocking=True)   # (B,)

            # CFG class dropout: replace some labels with the NULL class.
            drop_mask = torch.rand(labels.shape, device=device) < class_dropout_p
            labels = torch.where(drop_mask, torch.full_like(labels, CLASS_NULL), labels)

            B = y.shape[0]
            t = torch.randint(
                0, scheduler_train.config.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )
            noise = torch.randn_like(y)
            x_noisy = scheduler_train.add_noise(y, noise, t)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = unet(x_noisy, t, class_labels=labels).sample
            # Cast to fp32 *at the metric/loss boundary* -- avoids fp16 NaN spikes
            # under autocast on small batches (lesson learned from prior runs).
            loss = F.mse_loss(eps_pred.float(), noise.float())

            scaler.scale(loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
                lr_sched.step()
                optim.zero_grad(set_to_none=True)
                ema.update(unet)
                global_step += 1

            epoch_loss += float(loss.item())
            n_batches += 1

            if global_step > 0 and (global_step % log_every == 0):
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{lr_sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    import wandb
                    wandb.log(
                        {"train/loss": loss.item(),
                         "lr": lr_sched.get_last_lr()[0]},
                        step=global_step,
                    )

        avg_train_loss = epoch_loss / max(1, n_batches)

        # -------- Validation ------------------------------------------------
        unet.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                # No class dropout in validation.
                B = y.shape[0]
                t = torch.randint(
                    0, scheduler_train.config.num_train_timesteps, (B,),
                    device=device, dtype=torch.long,
                )
                noise = torch.randn_like(y)
                x_noisy = scheduler_train.add_noise(y, noise, t)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    eps_pred = unet(x_noisy, t, class_labels=labels).sample
                vloss = F.mse_loss(eps_pred.float(), noise.float())
                val_loss_sum += float(vloss.item())
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, n_val_batches)
        is_best = avg_val_loss < best_val
        if is_best:
            best_val = avg_val_loss

        print(f"[epoch {epoch + 1}] "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"best={best_val:.4f}{'  *' if is_best else ''}")
        if use_wandb:
            import wandb
            wandb.log(
                {"epoch": epoch + 1,
                 "train/epoch_loss": avg_train_loss,
                 "val/loss": avg_val_loss,
                 "val/best": best_val},
                step=global_step,
            )

        # -------- Sample grid (uses EMA weights) ----------------------------
        if (epoch + 1) % sample_every == 0:
            try:
                rgb = quick_class_samples(
                    unet=ema.ema_model,
                    scheduler_inf=scheduler_inf,
                    device=device,
                    image_size=image_size,
                    n_per_class=2,
                    cfg_scale=4.0,
                    num_steps=50,
                )
                save_sample_grid(
                    rgb,
                    paths["samples"] / f"epoch_{epoch + 1:04d}.png",
                    nrow=2,
                )
            except Exception as e:                # noqa: BLE001
                print(f"[warn] sampling failed: {e}")

        # -------- Checkpoint ------------------------------------------------
        if (epoch + 1) % ckpt_every == 0 or is_best:
            state = {
                "unet": unet.state_dict(),
                "ema": ema.state_dict(),
                "optim": optim.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "config": cfg,
            }
            ckpt_mgr.save(state, is_best=is_best)

    print("Training complete.")


if __name__ == "__main__":
    main()

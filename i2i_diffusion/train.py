"""
train.py
--------
Entry-point for training the Illumination Artifact Diffusion Pipeline.

Usage
-----
    python train.py --config configs/train_config.yaml

Optional resume
---------------
    python train.py --config configs/train_config.yaml \
                    --resume runs/illum_v1/checkpoints/epoch_0050.pt

Single-GPU DGX docker container
---------------------------------
The script uses Hugging Face Accelerate for device placement.
For single-GPU use you do NOT need to run `accelerate config` —
just call train.py directly.  Accelerate detects the single available
GPU automatically.

Mixed precision
---------------
Set training.mixed_precision to "fp16" (default) or "no" if you
encounter NaN losses on a specific model scale.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset    import UnpairedIlluminationDataset
from training.trainer import IlluminationDiffusionTrainer
from utils.metrics   import IlluminationMetrics

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(cfg: dict) -> Path:
    run_dir = Path(cfg["training"]["output_dir"]) / cfg["training"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    # save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    return run_dir


# ── lr scheduler wrapper ───────────────────────────────────────────────────────

def _build_lr_scheduler(optimizer, cfg: dict, num_training_steps: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR

    name     = cfg["training"]["lr_scheduler"]
    warmup   = cfg["training"]["warmup_steps"]

    # warmup lambda
    def warmup_fn(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return 1.0

    warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_fn)

    if name == "cosine":
        main_sched = CosineAnnealingLR(
            optimizer, T_max=num_training_steps - warmup, eta_min=1e-6
        )
    elif name == "linear":
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-6 / cfg["training"]["lr"],
            total_iters=num_training_steps - warmup,
        )
    else:  # constant
        main_sched = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # chain: warmup then main
    from torch.optim.lr_scheduler import SequentialLR
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, main_sched],
        milestones=[warmup],
    )


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    trainer:    IlluminationDiffusionTrainer,
    loader:     DataLoader,
    metrics_fn: IlluminationMetrics,
    cfg:        dict,
    epoch:      int,
) -> dict:
    """
    Run a lightweight evaluation pass over `loader`.
    Generates a small batch of samples and computes all metrics.
    """
    trainer.unet.eval()
    if trainer.controlnet:
        trainer.controlnet.eval()

    batch = next(iter(loader))

    source = batch["normal"].to(trainer.dev)
    edge   = batch["normal_edge"].to(trainer.dev)
    real_b = batch["over"].to(trainer.dev)
    real_c = batch["under"].to(trainer.dev)
    l_over  = batch["label_over"].to(trainer.dev)
    l_under = batch["label_under"].to(trainer.dev)

    # quick single-step samples for fast eval
    gen_over  = trainer._quick_sample(source, edge, l_over,  t_level=200)
    gen_under = trainer._quick_sample(source, edge, l_under, t_level=200)

    results = metrics_fn.evaluate_batch(
        gen_over, gen_under, source, real_b, real_c
    )

    # log image grid
    trainer.logger.log_image_grid(source, gen_over, gen_under, epoch)

    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main(config_path: str, resume: str | None = None) -> None:
    cfg = _load_config(config_path)
    tc  = cfg["training"]

    _set_seed(tc["seed"])

    # ── accelerator (handles single-GPU automatically) ─────────────────────
    accelerator = Accelerator(
        mixed_precision = tc["mixed_precision"],
        log_with        = None,   # we manage wandb ourselves
    )
    log.info(f"Device: {accelerator.device}  |  "
             f"Mixed precision: {tc['mixed_precision']}")

    run_dir = _make_run_dir(cfg)

    # ── dataset ────────────────────────────────────────────────────────────
    dc = cfg["data"]
    train_ds = UnpairedIlluminationDataset(
        normal_dir  = dc["normal_dir"],
        over_dir    = dc["over_dir"],
        under_dir   = dc["under_dir"],
        image_size  = dc["image_size"],
        augment     = True,
        seed        = tc["seed"],
    )
    val_ds = UnpairedIlluminationDataset(
        normal_dir  = dc["normal_dir"],
        over_dir    = dc["over_dir"],
        under_dir   = dc["under_dir"],
        image_size  = dc["image_size"],
        augment     = False,
        seed        = tc["seed"] + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = tc["batch_size"],
        shuffle     = True,
        num_workers = dc["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tc["batch_size"],
        shuffle     = False,
        num_workers = dc["num_workers"],
        pin_memory  = True,
    )

    # ── trainer ────────────────────────────────────────────────────────────
    trainer = IlluminationDiffusionTrainer(cfg, accelerator)

    # ── lr schedulers ─────────────────────────────────────────────────────
    num_training_steps = len(train_loader) * tc["num_epochs"]
    lr_sched_g = _build_lr_scheduler(trainer.opt_g, cfg, num_training_steps)
    lr_sched_d = _build_lr_scheduler(trainer.opt_d, cfg, num_training_steps)

    # ── accelerate: prepare models, optimisers, loaders ───────────────────
    (
        trainer.unet,
        trainer.disc_over,
        trainer.disc_under,
        trainer.opt_g,
        trainer.opt_d,
        train_loader,
        val_loader,
    ) = accelerator.prepare(
        trainer.unet,
        trainer.disc_over,
        trainer.disc_under,
        trainer.opt_g,
        trainer.opt_d,
        train_loader,
        val_loader,
    )
    if trainer.controlnet is not None:
        trainer.controlnet = accelerator.prepare(trainer.controlnet)

    # EMA shadow was deepcopied before accelerator.prepare() moved the live
    # model to GPU, so it is still on CPU.  Sync it now explicitly.
    if trainer.ema is not None:
        trainer.ema.to(accelerator.device)

    # ── metrics ────────────────────────────────────────────────────────────
    metrics_fn = IlluminationMetrics(accelerator.device)

    # ── optional resume ────────────────────────────────────────────────────
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(resume)
        log.info(f"Resuming from epoch {start_epoch}")

    # ── training loop ──────────────────────────────────────────────────────
    log.info(
        f"Starting training: {tc['num_epochs']} epochs, "
        f"{len(train_ds)} training samples, "
        f"batch size {tc['batch_size']}."
    )

    best_ssim = 0.0

    for epoch in range(start_epoch, tc["num_epochs"]):
        log.info(f"── Epoch {epoch + 1}/{tc['num_epochs']} ──")

        # train
        epoch_metrics = trainer.train_epoch(train_loader, epoch)
        trainer.logger.log_scalars(
            {f"epoch/{k}": v for k, v in epoch_metrics.items()},
            step=epoch,
        )
        lr_sched_g.step()
        lr_sched_d.step()

        # evaluation
        if (epoch + 1) % tc["eval_every"] == 0:
            val_metrics = run_eval(
                trainer, val_loader, metrics_fn, cfg, epoch
            )
            log.info(
                "Val metrics: "
                + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )
            trainer.logger.log_scalars(
                {f"val/{k}": v for k, v in val_metrics.items()},
                step=epoch,
            )

            # save best checkpoint (by structural SSIM of overexposed branch)
            ssim_over = val_metrics.get("ssim_grad_over", 0.0)
            if ssim_over > best_ssim:
                best_ssim = ssim_over
                ckpt_dir  = Path(run_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # write to a temp name then copy to best.pt atomically
                tmp_path  = ckpt_dir / "_best_tmp.pt"
                best_path = ckpt_dir / "best.pt"
                trainer.save_checkpoint(str(run_dir), epoch, filename="_best_tmp.pt")
                import shutil
                shutil.move(str(tmp_path), str(best_path))
                log.info(f"  ↑ New best SSIM {best_ssim:.4f} → saved best.pt")

        # periodic "last" checkpoint — delete previous last to save disk space
        if (epoch + 1) % tc["save_every"] == 0:
            ckpt_dir  = Path(run_dir) / "checkpoints"
            last_path = ckpt_dir / "last.pt"
            # remove previous last if it exists
            if last_path.exists():
                last_path.unlink()
            trainer.save_checkpoint(str(run_dir), epoch + 1, filename="last.pt")
            log.info(f"  Checkpoint saved → last.pt (epoch {epoch + 1})")

    trainer.logger.finish()
    log.info("Training complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/train_config.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint to resume from."
    )
    args = parser.parse_args()
    main(args.config, args.resume)
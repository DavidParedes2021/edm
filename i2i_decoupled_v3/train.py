"""
train.py
---------
Entry point for training the illumination diffusion model.

Usage
-----
# Standard training:
    python train.py --config configs/config.yaml

# Override data paths:
    python train.py --config configs/config.yaml \
        --normal_dir /path/to/normal \
        --over_dir   /path/to/over \
        --under_dir  /path/to/under \
        --output_dir ./outputs

# Resume from checkpoint:
    python train.py --config configs/config.yaml --resume outputs/checkpoints/latest.pt

# Generate synthetic dataset after training:
    python train.py --config configs/config.yaml --generate \
        --normal_dir /path/to/normal \
        --synth_output /path/to/synthetic_pairs
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from training.trainer import IlluminationTrainer
from utils.misc import load_checkpoint


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Illumination Diffusion Training")

    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--normal_dir", type=str, default=None, help="Override normal images path")
    p.add_argument("--over_dir",   type=str, default=None, help="Override overexposed images path")
    p.add_argument("--under_dir",  type=str, default=None, help="Override underexposed images path")
    p.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    p.add_argument("--resume",     type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--generate",   action="store_true",    help="Generate synthetic dataset (no training)")
    p.add_argument("--synth_output", type=str, default="./synthetic_pairs", help="Output dir for synthetic pairs")
    p.add_argument("--guidance_scale", type=float, default=None, help="Override CFG guidance scale at inference")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading + CLI overrides
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.normal_dir:
        cfg["data"]["normal_dir"]      = args.normal_dir
    if args.over_dir:
        cfg["data"]["overexposed_dir"] = args.over_dir
    if args.under_dir:
        cfg["data"]["underexposed_dir"] = args.under_dir
    if args.output_dir:
        cfg["checkpoint"]["output_dir"]      = args.output_dir
        cfg["checkpoint"]["checkpoint_dir"]  = os.path.join(args.output_dir, "checkpoints")
        cfg["checkpoint"]["samples_dir"]     = os.path.join(args.output_dir, "samples")
    if args.guidance_scale is not None:
        cfg["diffusion"]["classifier_free_guidance_scale"] = args.guidance_scale
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_cli_overrides(cfg, args)

    logger.info("=" * 60)
    logger.info("Illumination Diffusion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config:  {args.config}")
    logger.info(f"Normal:  {cfg['data']['normal_dir']}")
    logger.info(f"Over:    {cfg['data']['overexposed_dir']}")
    logger.info(f"Under:   {cfg['data']['underexposed_dir']}")
    logger.info(f"Output:  {cfg['checkpoint']['output_dir']}")

    trainer = IlluminationTrainer(cfg)

    # Resume
    if args.resume:
        ckpt = load_checkpoint(
            Path(args.resume),
            trainer.diffusion.model,
            trainer.ema,
            trainer.optimizer,
            trainer.scheduler,
            device=trainer.device,
        )
        trainer.start_epoch = ckpt.get("epoch", 0) + 1
        trainer.global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from epoch {trainer.start_epoch}")

    if args.generate:
        trainer.generate_synthetic_dataset(
            normal_dir     = cfg["data"]["normal_dir"],
            output_dir     = args.synth_output,
            ddim_steps     = cfg["diffusion"]["ddim_steps"],
            guidance_scale = cfg["diffusion"]["classifier_free_guidance_scale"],
            checkpoint     = "best",
        )
    else:
        trainer.train()


if __name__ == "__main__":
    main()

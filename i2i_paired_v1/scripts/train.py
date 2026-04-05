#!/usr/bin/env python
"""
scripts/train.py
Entry point for training the illumination diffusion model.

Usage:
  # Laptop smoke test (4GB GPU)
  python scripts/train.py --config configs/debug.yaml

  # Full training (DGX 16GB GPU)
  python scripts/train.py --config configs/train.yaml

  # Resume from last checkpoint
  python scripts/train.py --config configs/train.yaml --resume

  # Override any config key
  python scripts/train.py --config configs/train.yaml --set training.batch_size=2
"""

import argparse
import sys
import os
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config_utils import load_config
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Illumination Diffusion Model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/debug.yaml)"
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values. E.g. --set training.batch_size=2 model.base_channels=64"
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list):
    """Apply CLI key=value overrides to nested config dict."""
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring malformed override: {item}")
            continue
        key_path, value = item.split("=", 1)
        keys = key_path.strip().split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # Auto-type conversion
        final_key = keys[-1]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
        d[final_key] = value
        logger.info(f"Override: {key_path} = {value}")
    return cfg


def check_dataset_exists(cfg: dict):
    """Warn (don't crash) if dataset directories are missing — useful for CI."""
    import os
    data_cfg = cfg["data"]
    roots = [data_cfg.get("underexposed_root", ""), data_cfg.get("overexposed_root", "")]
    any_found = False
    for root in roots:
        if root and os.path.exists(root):
            any_found = True
    if not any_found:
        print(
            "\n[WARNING] Neither dataset root directory exists. "
            "Training will fail when trying to load data.\n"
            "Expected paths:\n"
            f"  {data_cfg.get('underexposed_root')}/train/underexposed/\n"
            f"  {data_cfg.get('underexposed_root')}/train/normal_frames/\n"
            f"  {data_cfg.get('overexposed_root')}/train/overexposed/\n"
            f"  {data_cfg.get('overexposed_root')}/train/normal_frames/\n"
            "\nCreate dummy data with: python scripts/create_dummy_data.py\n"
        )
    return any_found


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])

    # Print config summary
    print("=" * 60)
    print(f"  Illumination Diffusion Training")
    print(f"  Config       : {args.config}")
    print(f"  Experiment   : {cfg.get('experiment_name', 'N/A')}")
    print(f"  Image size   : {cfg['data']['image_size']}")
    print(f"  Base channels: {cfg['model']['base_channels']}")
    print(f"  Epochs       : {cfg['training']['num_epochs']}")
    print(f"  Batch size   : {cfg['training']['batch_size']}")
    print(f"  Mixed prec.  : {cfg['training'].get('mixed_precision', 'no')}")
    print(f"  Output dir   : {cfg.get('output_dir', 'outputs')}")
    print("=" * 60)

    import torch
    print(f"\nPyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM      : {total_mem:.1f} GB")
    print()

    # Dataset check
    check_dataset_exists(cfg)

    # Run training
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

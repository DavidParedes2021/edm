#!/usr/bin/env python3
# scripts/prepare_dataset.py
"""
Dataset preparation and validation script.

Reads dataset paths from the same config file used for training,
so there is a single source of truth for all paths.

Usage:
  # Validate your real dataset:
  python scripts/prepare_dataset.py --config configs/dgx_train.yaml

  # Create a tiny dummy dataset for smoke testing, then validate:
  python scripts/prepare_dataset.py --config configs/laptop_debug.yaml --create_dummy

  # Override paths without editing the config:
  python scripts/prepare_dataset.py --config configs/dgx_train.yaml \
      --normal_path /mnt/data/frames/normal \
      --over_path   /mnt/data/frames/over \
      --under_path  /mnt/data/frames/under
"""
import sys
import argparse
import logging
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_luminance_stats(img: Image.Image) -> dict:
    """Compute luminance statistics from a PIL image."""
    rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    return {
        "mean": float(y.mean()),
        "std":  float(y.std()),
        "min":  float(y.min()),
        "max":  float(y.max()),
        "p5":   float(np.percentile(y, 5)),
        "p95":  float(np.percentile(y, 95)),
    }


def validate_and_report(normal_path: str, over_path: str, under_path: str):
    """Scan and print statistics for all three splits."""
    splits = [
        ("normal", normal_path, lambda m: 0.3 < m < 0.7,  "0.3 < mean < 0.7"),
        ("over",   over_path,   lambda m: m > 0.5,        "mean > 0.5"),
        ("under",  under_path,  lambda m: m < 0.45,       "mean < 0.45"),
    ]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    all_ok = True
    for split_name, path_str, check_fn, check_desc in splits:
        d = Path(path_str)
        print(f"\n{'='*52}")
        print(f"  {split_name.upper():8s}: {d}")

        if not d.exists():
            logger.error(f"  MISSING — check '{split_name}_path' in your config")
            all_ok = False
            continue

        files = sorted([f for f in d.iterdir() if f.suffix.lower() in exts])
        if not files:
            logger.warning(f"  EMPTY — no images found")
            all_ok = False
            continue

        stats_all, sizes, corrupt = [], set(), []
        for f in tqdm(files, desc=f"  Scanning", leave=False):
            try:
                img = Image.open(f)
                sizes.add(img.size)
                stats_all.append(compute_luminance_stats(img))
            except Exception as e:
                corrupt.append((f.name, str(e)))

        means = [s["mean"] for s in stats_all]
        mean_lum = float(np.mean(means))
        ok = check_fn(mean_lum)
        all_ok = all_ok and ok

        print(f"  Images:          {len(files)}")
        print(f"  Corrupt:         {len(corrupt)}")
        print(f"  Unique sizes:    {len(sizes)} "
              f"{'(WARNING: mixed sizes — all will be resized)' if len(sizes) > 1 else 'OK'}")
        print(f"  Luminance mean:  {mean_lum:.3f} ± {float(np.std(means)):.3f}")
        print(f"  Lum p5/p95:      [{min(s['p5'] for s in stats_all):.3f}, "
              f"{max(s['p95'] for s in stats_all):.3f}]")
        print(f"  Exposure check ({check_desc}): {'✓ PASS' if ok else '✗ FAIL'}")

        if corrupt:
            print(f"  CORRUPT files (first 5):")
            for name, err in corrupt[:5]:
                print(f"    {name}: {err}")

    print(f"\n{'='*52}")
    print("Expected distribution for good training:")
    print("  normal: mean Y ≈ 0.40–0.55")
    print("  over:   mean Y ≈ 0.60–0.85")
    print("  under:  mean Y ≈ 0.15–0.40")
    print(f"\nOverall: {'✓ READY FOR TRAINING' if all_ok else '✗ ISSUES FOUND — review above'}")


def create_dummy_dataset(normal_path: str, over_path: str, under_path: str,
                         n_normal: int = 20, n_over: int = 10, n_under: int = 20):
    """
    Create a tiny dummy dataset for smoke testing on laptop.
    Images are created at the exact paths specified in the config.
    """
    rng = np.random.default_rng(42)

    for path_str, count, exposure, label in [
        (normal_path, n_normal, 0.50, "normal"),
        (over_path,   n_over,   0.75, "over"),
        (under_path,  n_under,  0.25, "under"),
    ]:
        d = Path(path_str)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_arr = np.clip(rng.uniform(0, 1, (128, 128, 3)) * exposure * 2, 0, 1)
            img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
            img_pil.save(d / f"frame_{i:04d}.png")
        logger.info(f"  Created {count} dummy {label} frames → {d}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate or create dataset using paths from a training config."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML (e.g. configs/dgx_train.yaml)"
    )
    parser.add_argument(
        "--create_dummy", action="store_true",
        help="Create a small dummy dataset at the configured paths (for smoke testing)"
    )
    # Optional per-call overrides — useful when testing with a temp location
    parser.add_argument("--normal_path", type=str, default=None,
                        help="Override data.normal_path from config")
    parser.add_argument("--over_path",   type=str, default=None,
                        help="Override data.over_path from config")
    parser.add_argument("--under_path",  type=str, default=None,
                        help="Override data.under_path from config")
    parser.add_argument("--n_normal", type=int, default=20)
    parser.add_argument("--n_over",   type=int, default=10)
    parser.add_argument("--n_under",  type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]

    # CLI overrides take precedence over config values
    normal_path = args.normal_path or dcfg["normal_path"]
    over_path   = args.over_path   or dcfg["over_path"]
    under_path  = args.under_path  or dcfg["under_path"]

    logger.info(f"Config:      {args.config}")
    logger.info(f"normal_path: {normal_path}")
    logger.info(f"over_path:   {over_path}")
    logger.info(f"under_path:  {under_path}")

    if args.create_dummy:
        logger.info("Creating dummy dataset...")
        create_dummy_dataset(
            normal_path=normal_path,
            over_path=over_path,
            under_path=under_path,
            n_normal=args.n_normal,
            n_over=args.n_over,
            n_under=args.n_under,
        )

    validate_and_report(normal_path, over_path, under_path)


if __name__ == "__main__":
    main()
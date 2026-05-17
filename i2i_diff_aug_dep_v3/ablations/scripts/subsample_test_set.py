#!/usr/bin/env python3
"""
subsample_test_set.py — Pick N test images deterministically (sorted +
seeded random.sample) and copy them into a fresh directory so every
ablation evaluator scores the SAME held-out subset.

Idempotent: if the destination already contains the right number of files
the script exits without changes. Pass --force to overwrite.
"""
import argparse
import random
import shutil
from pathlib import Path

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"--src not a directory: {src}")

    files = []
    for ext in EXTS:
        files.extend(sorted(src.glob(f"*{ext}")))
        files.extend(sorted(src.glob(f"*{ext.upper()}")))
    files = sorted(set(files))  # de-dup case-insensitive matches on Windows
    if not files:
        raise SystemExit(f"No images found in {src}")

    n = min(args.n, len(files))
    rng = random.Random(args.seed)
    picked = rng.sample(files, n)

    if dst.exists() and not args.force:
        existing = sum(1 for ext in EXTS for _ in dst.glob(f"*{ext}"))
        if existing == n:
            print(f"[skip] {dst} already has {n} files; pass --force to overwrite.")
            return
    if dst.exists() and args.force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for src_path in picked:
        shutil.copy2(src_path, dst / src_path.name)
    print(f"[ok] copied {n} images from {src} → {dst} (seed={args.seed})")


if __name__ == "__main__":
    main()

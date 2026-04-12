#!/usr/bin/env python3
"""
generate_pairs.py — Create paired training data for the diffusion model.

Runs exposure_augment.py on every normal frame to produce:
    data/pairs/overexposed/   (target L channels saved as .npy)
    data/pairs/underexposed/  (target L channels saved as .npy)
    data/pairs/normal/        (source L channels + AB channels as .npy)

We store raw LAB arrays (not RGB images) so the diffusion model trains
directly on L channels without repeated colour-space conversions.

Usage:
    python generate_pairs.py \
        --normal_dir ./data/normal \
        --output_dir ./data/pairs \
        --strength 0.85 \
        --shift_magnitude 50.0

    # Generate multiple augmentation variants per image for more diversity:
    python generate_pairs.py \
        --normal_dir ./data/normal \
        --output_dir ./data/pairs \
        --num_variants 3
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from exposure_augment import rgb_to_lab, augment_exposure


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def generate_pairs(
    normal_dir: str,
    output_dir: str,
    strength: float = 0.85,
    shift_magnitude: float = 50.0,
    cluster_expand_px: int = 60,
    falloff_sigma: float = 30.0,
    bright_percentile: float = 85.0,
    dark_percentile: float = 15.0,
    texture_sigma: float = 3.0,
    num_variants: int = 1,
):
    normal_path = Path(normal_dir)
    files = sorted(p for p in normal_path.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in {normal_dir}")

    # output structure
    out = Path(output_dir)
    (out / "normal").mkdir(parents=True, exist_ok=True)
    (out / "overexposed").mkdir(parents=True, exist_ok=True)
    (out / "underexposed").mkdir(parents=True, exist_ok=True)

    # define parameter variations for multi-variant generation
    rng = np.random.RandomState(42)
    variant_params = []
    for v in range(num_variants):
        if v == 0:
            # variant 0 = exact defaults
            variant_params.append({
                "strength": strength,
                "shift_magnitude": shift_magnitude,
                "cluster_expand_px": cluster_expand_px,
                "falloff_sigma": falloff_sigma,
                "bright_percentile": bright_percentile,
                "dark_percentile": dark_percentile,
                "texture_sigma": texture_sigma,
            })
        else:
            # subsequent variants add controlled randomness
            variant_params.append({
                "strength": np.clip(strength + rng.uniform(-0.15, 0.15), 0.4, 1.0),
                "shift_magnitude": np.clip(shift_magnitude + rng.uniform(-15, 15), 20, 70),
                "cluster_expand_px": int(np.clip(cluster_expand_px + rng.randint(-20, 20), 30, 100)),
                "falloff_sigma": np.clip(falloff_sigma + rng.uniform(-10, 10), 15, 50),
                "bright_percentile": np.clip(bright_percentile + rng.uniform(-10, 5), 70, 92),
                "dark_percentile": np.clip(dark_percentile + rng.uniform(-5, 10), 8, 30),
                "texture_sigma": texture_sigma,
            })

    print(f"[Pair Generation] {len(files)} images × {num_variants} variant(s) "
          f"= {len(files) * num_variants} pairs per domain")

    count = 0
    for fpath in tqdm(files, desc="Generating pairs"):
        img_rgb = np.array(Image.open(fpath).convert("RGB"))
        lab = rgb_to_lab(img_rgb)

        for v, params in enumerate(variant_params):
            suffix = f"_v{v}" if num_variants > 1 else ""
            stem = fpath.stem + suffix

            # source: normal L and AB
            # (save once per image, not per variant — AB doesn't change)
            if v == 0:
                np.save(str(out / "normal" / f"{fpath.stem}.npy"), lab)

            # target: overexposed
            over_rgb = augment_exposure(img_rgb, mode="overexposed", **params)
            over_lab = rgb_to_lab(over_rgb)
            np.save(str(out / "overexposed" / f"{stem}.npy"), over_lab[..., 0])  # L only

            # target: underexposed
            under_rgb = augment_exposure(img_rgb, mode="underexposed", **params)
            under_lab = rgb_to_lab(under_rgb)
            np.save(str(out / "underexposed" / f"{stem}.npy"), under_lab[..., 0])  # L only

            count += 1

    print(f"[Done] {count} pairs generated in {output_dir}")
    print(f"  Normal LAB arrays:     {out / 'normal'}")
    print(f"  Overexposed L arrays:  {out / 'overexposed'}")
    print(f"  Underexposed L arrays: {out / 'underexposed'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/pairs")
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--shift_magnitude", type=float, default=50.0)
    parser.add_argument("--cluster_expand_px", type=int, default=60)
    parser.add_argument("--falloff_sigma", type=float, default=30.0)
    parser.add_argument("--bright_percentile", type=float, default=85.0)
    parser.add_argument("--dark_percentile", type=float, default=15.0)
    parser.add_argument("--texture_sigma", type=float, default=3.0)
    parser.add_argument("--num_variants", type=int, default=1,
                        help="Number of augmentation variants per image (more = more diversity)")
    args = parser.parse_args()

    generate_pairs(
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        strength=args.strength,
        shift_magnitude=args.shift_magnitude,
        cluster_expand_px=args.cluster_expand_px,
        falloff_sigma=args.falloff_sigma,
        bright_percentile=args.bright_percentile,
        dark_percentile=args.dark_percentile,
        texture_sigma=args.texture_sigma,
        num_variants=args.num_variants,
    )


if __name__ == "__main__":
    main()

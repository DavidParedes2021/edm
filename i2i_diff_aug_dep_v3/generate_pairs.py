#!/usr/bin/env python3
"""
generate_pairs.py — Create paired training data for the depth-aware
conditional diffusion model.

For every normal frame, run depth_augment.depth_aware_augment to produce:
    data/pairs/normal/       full LAB arrays                (one per image)
    data/pairs/depth/        depth maps in [0, 1]            (one per image)
    data/pairs/overexposed/  augmented L channels            (N variants/image)
    data/pairs/underexposed/ augmented L channels            (N variants/image)

Requires depth maps to have been precomputed by depth_estimator.py
(or they can be computed inline with --compute_depth_inline).

Usage:
    python depth_estimator.py --input_dir ./data/normal --output_dir ./data/depth

    python generate_pairs.py \\
        --normal_dir ./data/normal \\
        --depth_dir  ./data/depth \\
        --output_dir ./data/pairs \\
        --num_variants 3
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from exposure_augment import rgb_to_lab
from depth_augment import depth_aware_augment


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _load_depth(depth_dir: Path, stem: str, target_shape: tuple) -> np.ndarray:
    """Load depth .npy file, resize if needed."""
    depth_path = depth_dir / f"{stem}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(
            f"Depth map missing for {stem}. Run depth_estimator.py first."
        )
    d = np.load(str(depth_path)).astype(np.float32)
    if d.shape != target_shape:
        # resize via PIL
        img = Image.fromarray(d, mode="F").resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR
        )
        d = np.array(img, dtype=np.float32)
    # clamp to [0, 1] in case rounding spilled
    return np.clip(d, 0.0, 1.0)


def generate_pairs(
    normal_dir: str,
    depth_dir: str,
    output_dir: str,
    strength: float = 1.0,
    shift_magnitude_over: float = 65.0,
    shift_magnitude_under: float = 80.0,
    gamma_over: float = 2.0,
    gamma_under: float = 2.5,
    depth_blend: float = 1.0,
    mask_strategy: str = "luminance",
    hybrid_blend: float = 0.5,
    vignette_threshold: float = 5.0,
    num_variants: int = 1,
    seed: int = 42,
):
    normal_path = Path(normal_dir)
    depth_path = Path(depth_dir)
    files = sorted(p for p in normal_path.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in {normal_dir}")

    out = Path(output_dir)
    (out / "normal").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "overexposed").mkdir(parents=True, exist_ok=True)
    (out / "underexposed").mkdir(parents=True, exist_ok=True)

    # variant parameter jitter
    rng = np.random.RandomState(seed)
    variant_params = []
    for v in range(num_variants):
        if v == 0:
            variant_params.append({
                "strength": strength,
                "gamma_over": gamma_over,
                "gamma_under": gamma_under,
                "depth_blend": depth_blend,
            })
        else:
            variant_params.append({
                "strength": float(np.clip(strength + rng.uniform(-0.05, 0.0), 0.85, 1.0)),
                "gamma_over": float(np.clip(gamma_over + rng.uniform(-0.3, 0.5), 1.0, 2.5)),
                "gamma_under": float(np.clip(gamma_under + rng.uniform(-0.4, 0.5), 1.3, 3.0)),
                "depth_blend": float(np.clip(depth_blend + rng.uniform(-0.1, 0.0), 0.85, 1.0)),
            })

    print(f"[Pair Generation] {len(files)} images × {num_variants} variants "
          f"= {len(files) * num_variants} pairs per domain")
    for i, vp in enumerate(variant_params):
        print(f"  variant {i}: {vp}")

    count = 0
    for fpath in tqdm(files, desc="Generating pairs"):
        img_rgb = np.array(Image.open(fpath).convert("RGB"))
        H, W = img_rgb.shape[:2]

        try:
            depth = _load_depth(depth_path, fpath.stem, (H, W))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # save normal LAB once per image
        lab = rgb_to_lab(img_rgb)
        np.save(str(out / "normal" / f"{fpath.stem}.npy"), lab.astype(np.float32))
        np.save(str(out / "depth"  / f"{fpath.stem}.npy"), depth.astype(np.float16))

        for v, params in enumerate(variant_params):
            suffix = f"_v{v}" if num_variants > 1 else ""
            stem = fpath.stem + suffix

            common = dict(
                mask_strategy=mask_strategy,
                hybrid_blend=hybrid_blend,
                vignette_threshold=vignette_threshold,
            )

            # ── overexposed ──
            over_rgb = depth_aware_augment(
                img_rgb, depth,
                mode="overexposed",
                shift_magnitude=shift_magnitude_over,
                **params, **common,
            )
            over_lab = rgb_to_lab(over_rgb)
            np.save(str(out / "overexposed" / f"{stem}.npy"),
                    over_lab[..., 0].astype(np.float32))

            # ── underexposed ──
            under_rgb = depth_aware_augment(
                img_rgb, depth,
                mode="underexposed",
                shift_magnitude=shift_magnitude_under,
                **params, **common,
            )
            under_lab = rgb_to_lab(under_rgb)
            np.save(str(out / "underexposed" / f"{stem}.npy"),
                    under_lab[..., 0].astype(np.float32))

            count += 1

    print(f"[Done] {count} pairs per domain generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir", type=str, required=True)
    parser.add_argument("--depth_dir",  type=str, required=True,
                        help="Directory with .npy depth maps from depth_estimator.py")
    parser.add_argument("--output_dir", type=str, default="./data/pairs")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--shift_magnitude_over",  type=float, default=65.0)
    parser.add_argument("--shift_magnitude_under", type=float, default=80.0)
    parser.add_argument("--gamma_over",  type=float, default=2.0,
                        help="Higher = sharper peak on bright pixels (over).")
    parser.add_argument("--gamma_under", type=float, default=2.5,
                        help="Higher = sharper peak on dark pixels (under).")
    parser.add_argument("--depth_blend", type=float, default=1.0,
                        help="Used only when --mask_strategy=depth: depth vs cluster mix.")
    parser.add_argument("--mask_strategy", type=str, default="luminance",
                        choices=["luminance", "depth", "hybrid"],
                        help="luminance: use L itself as the cavity/foreground indicator "
                             "(default; robust for endoscopy). depth: use depth_estimator output. "
                             "hybrid: weighted blend of both.")
    parser.add_argument("--hybrid_blend", type=float, default=0.5,
                        help="Used only when --mask_strategy=hybrid: weight on luminance vs depth.")
    parser.add_argument("--vignette_threshold", type=float, default=5.0,
                        help="L threshold (0-100) below which a pixel is treated as camera "
                             "vignette and excluded from all augmentation masks.")
    parser.add_argument("--num_variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_pairs(
        normal_dir=args.normal_dir,
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        strength=args.strength,
        shift_magnitude_over=args.shift_magnitude_over,
        shift_magnitude_under=args.shift_magnitude_under,
        gamma_over=args.gamma_over,
        gamma_under=args.gamma_under,
        depth_blend=args.depth_blend,
        mask_strategy=args.mask_strategy,
        hybrid_blend=args.hybrid_blend,
        vignette_threshold=args.vignette_threshold,
        num_variants=args.num_variants,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

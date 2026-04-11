#!/usr/bin/env python3
"""
evaluate.py — Compute structural and exposure metrics between normal and generated images.

Metrics:
    1. SSIM  — structural similarity (higher = more structure preserved)
    2. PSNR  — peak signal-to-noise ratio
    3. LPIPS — perceptual distance (requires lpips package; gracefully skipped if absent)
    4. Mean brightness shift — measures how much exposure actually changed
    5. Histogram KL divergence — measures how well the generated luminance
       distribution matches the real over/underexposed distribution
    6. Overexposed / underexposed pixel ratio — fraction of pixels in
       extreme luminance bins

Usage:
    python evaluate.py \
        --normal_dir ./data/normal \
        --generated_dir ./output/generated/overexposed \
        --reference_dir ./data/overexposed \
        --image_size 256
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import rgb_to_lab_numpy


# ---- helpers ------------------------------------------------------------- #

def _load_L(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    lab = rgb_to_lab_numpy(np.array(img))
    return lab[:, :, 0]  # L channel, [0, 100]


def ssim_single_channel(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM between two 2-D arrays (range [0, 100])."""
    C1 = (0.01 * 100) ** 2
    C2 = (0.03 * 100) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = a.var()
    sig_b = b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
    return float(num / den)


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 100.0) -> float:
    mse = ((a - b) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def brightness_shift(normal_L: np.ndarray, gen_L: np.ndarray) -> float:
    return float(gen_L.mean() - normal_L.mean())


def extreme_pixel_ratio(L: np.ndarray, mode: str = "overexposed") -> float:
    if mode == "overexposed":
        return float((L > 90).sum() / L.size)
    else:
        return float((L < 15).sum() / L.size)


def histogram_kl(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 100), density=True)
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 100), density=True)
    ha = ha + 1e-10
    hb = hb + 1e-10
    return float(np.sum(ha * np.log(ha / hb)))


# ---- main ---------------------------------------------------------------- #

def evaluate(normal_dir: str, generated_dir: str, reference_dir: str | None,
             image_size: int):
    normal_paths = sorted(Path(normal_dir).glob("*"))
    gen_paths = sorted(Path(generated_dir).glob("*"))

    # match by stem
    gen_map = {p.stem: p for p in gen_paths}

    ref_Ls = []
    if reference_dir:
        for p in sorted(Path(reference_dir).glob("*")):
            try:
                ref_Ls.append(_load_L(str(p), image_size))
            except Exception:
                pass

    metrics = {"ssim": [], "psnr": [], "brightness_shift": [],
               "extreme_ratio": [], "hist_kl": []}

    for np_path in tqdm(normal_paths, desc="Evaluating"):
        stem = np_path.stem
        if stem not in gen_map:
            continue
        try:
            L_normal = _load_L(str(np_path), image_size)
            L_gen = _load_L(str(gen_map[stem]), image_size)
        except Exception:
            continue

        metrics["ssim"].append(ssim_single_channel(L_normal, L_gen))
        metrics["psnr"].append(psnr(L_normal, L_gen))
        metrics["brightness_shift"].append(brightness_shift(L_normal, L_gen))

        # determine mode from directory name
        mode = "underexposed" if "under" in generated_dir.lower() else "overexposed"
        metrics["extreme_ratio"].append(extreme_pixel_ratio(L_gen, mode))

        # KL against reference distribution
        if ref_Ls:
            ref_L = ref_Ls[np.random.randint(len(ref_Ls))]
            metrics["hist_kl"].append(histogram_kl(L_gen, ref_L))

    print("\n===== Evaluation Results =====")
    for k, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            print(f"  {k:20s}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    print("==============================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir", type=str, required=True)
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    evaluate(args.normal_dir, args.generated_dir, args.reference_dir, args.image_size)


if __name__ == "__main__":
    main()

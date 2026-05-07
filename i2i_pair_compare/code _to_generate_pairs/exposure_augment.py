#!/usr/bin/env python3
"""
exposure_augment.py — Deterministic cluster-based exposure augmentation.

Replaces the diffusion SDEdit inference with a physics-informed approach:
1. Convert to LAB, extract L channel
2. Extract high-pass texture detail (L - blur(L))
3. Identify seed clusters of bright/dark pixels via thresholding + morphology
4. Expand clusters with distance-weighted smooth falloff
5. Apply aggressive luminance shift to the low-frequency envelope
6. Re-inject the original high-pass texture
7. Recombine with untouched A,B channels → RGB

This guarantees:
- Texture is pixel-identical to the original (high-pass preserved)
- Colour is identical (A,B untouched)
- Exposure changes are strong and spatially coherent (cluster-based)
- Works at any resolution (no model, no GPU needed)

Usage:
    python exposure_augment.py --input_dir ./data/normal --output_dir ./output/augmented
    python exposure_augment.py --input_dir ./data/normal --output_dir ./output/augmented \
        --mode overexposed --strength 0.8 --cluster_expand_px 60
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label as connected_components,
)
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Colour-space conversions (pure NumPy, no OpenCV dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB → float32 HWC LAB. L∈[0,100], A∈[-128,127], B∈[-128,127]."""
    x = img.astype(np.float32) / 255.0
    mask = x > 0.04045
    x = np.where(mask, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = x @ mat.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    A = 500.0 * (f[..., 0] - f[..., 1])
    B = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 HWC LAB → uint8 HWC RGB."""
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=-1).astype(np.float32)
    mat_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    rgb = xyz @ mat_inv.T
    rgb = np.clip(rgb, 0.0, None)
    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055, 12.92 * rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Core exposure augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_luminance(L: np.ndarray, texture_sigma: float = 3.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Split L into low-frequency envelope + high-pass texture.

    Args:
        L: luminance channel, [0, 100]
        texture_sigma: Gaussian sigma for the split. Larger = more of the
                       image is considered "texture" and preserved.

    Returns:
        (L_low, L_high) where L = L_low + L_high
    """
    L_low = gaussian_filter(L.astype(np.float64), sigma=texture_sigma).astype(np.float32)
    L_high = L - L_low
    return L_low, L_high


def find_seed_clusters(
    L: np.ndarray,
    mode: str,
    bright_threshold_percentile: float = 85.0,
    dark_threshold_percentile: float = 15.0,
    min_cluster_pixels: int = 50,
) -> np.ndarray:
    """Find seed regions of already-bright or already-dark pixels.

    Uses adaptive percentile thresholds on the image's own distribution,
    then cleans with morphological operations and filters by size.

    Returns:
        Binary mask of seed clusters (True = seed pixel).
    """
    if mode == "overexposed":
        threshold = np.percentile(L, bright_threshold_percentile)
        # also require absolute brightness (at least mid-range)
        threshold = max(threshold, 45.0)
        seed_mask = L >= threshold
    elif mode == "underexposed":
        threshold = np.percentile(L, dark_threshold_percentile)
        # also require absolute darkness
        threshold = min(threshold, 55.0)
        seed_mask = L <= threshold
    else:
        raise ValueError(f"mode must be 'overexposed' or 'underexposed', got {mode}")

    # morphological cleanup: close small gaps, remove noise
    struct = np.ones((5, 5), dtype=bool)
    seed_mask = binary_erosion(seed_mask, structure=struct, iterations=1)
    seed_mask = binary_dilation(seed_mask, structure=struct, iterations=2)

    # filter out tiny clusters
    labelled, n_clusters = connected_components(seed_mask)
    for i in range(1, n_clusters + 1):
        cluster_size = (labelled == i).sum()
        if cluster_size < min_cluster_pixels:
            seed_mask[labelled == i] = False

    return seed_mask


def build_exposure_map(
    seed_mask: np.ndarray,
    L: np.ndarray,
    mode: str,
    expand_radius_px: int = 60,
    falloff_sigma: float = 30.0,
    core_strength: float = 1.0,
) -> np.ndarray:
    """Build a smooth [0, 1] exposure influence map from seed clusters.

    The map is 1.0 at seed cores and falls off smoothly with distance.
    Pixels far from any seed cluster get 0 influence.

    Args:
        seed_mask: binary mask of seed clusters
        L: original luminance (used to modulate influence by local brightness)
        mode: 'overexposed' or 'underexposed'
        expand_radius_px: maximum expansion distance in pixels
        falloff_sigma: Gaussian sigma controlling how fast influence decays
        core_strength: strength multiplier at seed cores

    Returns:
        Float32 map in [0, 1] — the blending weight for exposure shift.
    """
    if not seed_mask.any():
        # no clusters found — create a mild global shift from the brightest/darkest region
        if mode == "overexposed":
            influence = (L - L.min()) / (L.max() - L.min() + 1e-8)
            influence = np.power(influence, 3.0)  # concentrate on brightest areas
        else:
            influence = 1.0 - (L - L.min()) / (L.max() - L.min() + 1e-8)
            influence = np.power(influence, 3.0)  # concentrate on darkest areas
        return (influence * core_strength * 0.5).astype(np.float32)

    # distance from nearest seed pixel
    inv_mask = ~seed_mask
    dist = distance_transform_edt(inv_mask).astype(np.float32)

    # smooth falloff: Gaussian decay from seed boundary
    influence = np.exp(-0.5 * (dist / falloff_sigma) ** 2)

    # hard cutoff beyond expand_radius
    influence[dist > expand_radius_px] = 0.0

    # seed cores get full strength
    influence[seed_mask] = core_strength

    # modulate by local brightness affinity:
    # for overexposure, brighter neighbours expand more easily
    # for underexposure, darker neighbours expand more easily
    L_norm = (L - L.min()) / (L.max() - L.min() + 1e-8)
    if mode == "overexposed":
        affinity = np.power(L_norm, 0.5)  # bright pixels are more receptive
    else:
        affinity = np.power(1.0 - L_norm, 0.5)  # dark pixels more receptive

    # only apply affinity outside the core seeds
    blended = influence.copy()
    blended[inv_mask] *= affinity[inv_mask]

    # smooth the whole map to remove hard edges
    blended = gaussian_filter(blended, sigma=falloff_sigma * 0.3)

    # re-normalise to [0, core_strength]
    if blended.max() > 0:
        blended = blended / blended.max() * core_strength

    return blended.astype(np.float32)


def apply_exposure_shift(
    L_low: np.ndarray,
    exposure_map: np.ndarray,
    mode: str,
    shift_magnitude: float = 40.0,
    clip_target_over: float = 100.0,
    clip_target_under: float = 0.0,
) -> np.ndarray:
    """Apply luminance shift to the low-frequency envelope.

    Args:
        L_low: low-frequency luminance envelope
        exposure_map: [0,1] influence map
        mode: 'overexposed' or 'underexposed'
        shift_magnitude: maximum L shift in absolute units (0-100 scale)
        clip_target_over: L value for fully overexposed pixels
        clip_target_under: L value for fully underexposed pixels

    Returns:
        Modified L_low with exposure shift applied.
    """
    L_shifted = L_low.copy()

    if mode == "overexposed":
        # push toward clipping white — stronger where influence is higher
        # use a non-linear curve so the effect accelerates near the core
        shift = exposure_map * shift_magnitude
        # additionally, pixels already bright get pushed harder toward clip
        brightness_boost = (L_low / 100.0) ** 0.5  # brighter pixels saturate faster
        shift *= (0.5 + 0.5 * brightness_boost)
        L_shifted = L_shifted + shift
        # core saturation: where exposure_map > 0.9, push strongly toward clip
        core = exposure_map > 0.85
        if core.any():
            L_shifted[core] = L_shifted[core] * 0.3 + clip_target_over * 0.7

    elif mode == "underexposed":
        shift = exposure_map * shift_magnitude
        darkness_pull = (1.0 - L_low / 100.0) ** 0.5
        shift *= (0.5 + 0.5 * darkness_pull)
        L_shifted = L_shifted - shift
        core = exposure_map > 0.85
        if core.any():
            L_shifted[core] = L_shifted[core] * 0.3 + clip_target_under * 0.7

    return L_shifted


def augment_exposure(
    img_rgb: np.ndarray,
    mode: str = "overexposed",
    strength: float = 0.8,
    texture_sigma: float = 3.0,
    cluster_expand_px: int = 60,
    falloff_sigma: float = 30.0,
    bright_percentile: float = 85.0,
    dark_percentile: float = 15.0,
    min_cluster_px: int = 50,
    shift_magnitude: float = 45.0,
) -> np.ndarray:
    """Full pipeline: RGB in → exposure-augmented RGB out.

    Args:
        img_rgb: uint8 HWC RGB input (any resolution)
        mode: 'overexposed' or 'underexposed'
        strength: overall effect strength in [0, 1]. 1.0 = full effect.
        texture_sigma: sigma for luminance decomposition
        cluster_expand_px: how far (pixels) clusters expand
        falloff_sigma: smoothness of expansion falloff
        bright_percentile: percentile threshold for overexposure seeds
        dark_percentile: percentile threshold for underexposure seeds
        min_cluster_px: minimum seed cluster size
        shift_magnitude: max L-channel shift (0–100 scale)

    Returns:
        uint8 HWC RGB with exposure augmentation applied.
    """
    # --- Step 1: Convert to LAB ---
    lab = rgb_to_lab(img_rgb)
    L = lab[..., 0].copy()      # [0, 100]
    A = lab[..., 1].copy()      # untouched
    B = lab[..., 2].copy()      # untouched

    # scale parameters by image size (reference: 512px)
    scale = max(img_rgb.shape[0], img_rgb.shape[1]) / 512.0
    scaled_expand = int(cluster_expand_px * scale)
    scaled_falloff = falloff_sigma * scale
    scaled_texture_sigma = texture_sigma * scale
    scaled_min_cluster = int(min_cluster_px * scale * scale)  # area scales quadratically

    # --- Step 2: Decompose luminance ---
    L_low, L_high = decompose_luminance(L, texture_sigma=scaled_texture_sigma)

    # --- Step 3: Find seed clusters ---
    seed_mask = find_seed_clusters(
        L, mode,
        bright_threshold_percentile=bright_percentile,
        dark_threshold_percentile=dark_percentile,
        min_cluster_pixels=scaled_min_cluster,
    )

    # --- Step 4: Build exposure influence map ---
    exposure_map = build_exposure_map(
        seed_mask, L, mode,
        expand_radius_px=scaled_expand,
        falloff_sigma=scaled_falloff,
        core_strength=strength,
    )

    # --- Step 5: Shift the low-frequency envelope only ---
    L_low_shifted = apply_exposure_shift(
        L_low, exposure_map, mode,
        shift_magnitude=shift_magnitude,
    )

    # --- Step 6: Recombine: shifted envelope + original texture ---
    L_new = L_low_shifted + L_high

    # clip to valid LAB range
    L_new = np.clip(L_new, 0.0, 100.0).astype(np.float32)

    # --- Step 7: Reconstruct LAB → RGB ---
    lab_new = np.stack([L_new, A, B], axis=-1)
    rgb_out = lab_to_rgb(lab_new)

    return rgb_out


# ═══════════════════════════════════════════════════════════════════════════════
# Batch processing CLI
# ═══════════════════════════════════════════════════════════════════════════════

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def process_directory(
    input_dir: str,
    output_dir: str,
    mode: str = "both",
    strength: float = 0.8,
    cluster_expand_px: int = 60,
    falloff_sigma: float = 30.0,
    shift_magnitude: float = 45.0,
    bright_percentile: float = 85.0,
    dark_percentile: float = 15.0,
    texture_sigma: float = 3.0,
):
    input_path = Path(input_dir)
    files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"No images found in {input_dir}")
        return

    modes = []
    if mode in ("overexposed", "both"):
        modes.append("overexposed")
    if mode in ("underexposed", "both"):
        modes.append("underexposed")

    for m in modes:
        out_dir = Path(output_dir) / m
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{m.upper()}] Processing {len(files)} images → {out_dir}")
        for fpath in tqdm(files, desc=m):
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                result = augment_exposure(
                    img, mode=m,
                    strength=strength,
                    texture_sigma=texture_sigma,
                    cluster_expand_px=cluster_expand_px,
                    falloff_sigma=falloff_sigma,
                    shift_magnitude=shift_magnitude,
                    bright_percentile=bright_percentile,
                    dark_percentile=dark_percentile,
                )
                Image.fromarray(result).save(str(out_dir / f"{fpath.stem}.png"))
            except Exception as e:
                print(f"  [ERROR] {fpath.name}: {e}")

    print(f"\n[Done] Output in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-based exposure augmentation for endoscopy frames"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of normal frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for augmented frames")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["overexposed", "underexposed", "both"])
    parser.add_argument("--strength", type=float, default=0.8,
                        help="Overall effect strength [0-1] (default: 0.8)")
    parser.add_argument("--cluster_expand_px", type=int, default=60,
                        help="Cluster expansion radius in pixels (at 512px ref)")
    parser.add_argument("--falloff_sigma", type=float, default=30.0,
                        help="Gaussian falloff sigma for expansion")
    parser.add_argument("--shift_magnitude", type=float, default=45.0,
                        help="Max L-channel shift magnitude [0-100]")
    parser.add_argument("--bright_percentile", type=float, default=85.0,
                        help="Brightness percentile for overexposure seeds")
    parser.add_argument("--dark_percentile", type=float, default=15.0,
                        help="Darkness percentile for underexposure seeds")
    parser.add_argument("--texture_sigma", type=float, default=3.0,
                        help="Gaussian sigma for texture decomposition")
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        strength=args.strength,
        cluster_expand_px=args.cluster_expand_px,
        falloff_sigma=args.falloff_sigma,
        shift_magnitude=args.shift_magnitude,
        bright_percentile=args.bright_percentile,
        dark_percentile=args.dark_percentile,
        texture_sigma=args.texture_sigma,
    )


if __name__ == "__main__":
    main()

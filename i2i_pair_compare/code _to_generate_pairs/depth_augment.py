#!/usr/bin/env python3
"""
depth_augment.py — Depth-aware exposure augmentation.

Wraps exposure_augment.augment_exposure and replaces the cluster-based
exposure mask with a DEPTH-DRIVEN mask, retaining a small contribution
from the luminance-cluster mask for stochastic variation.

Why depth:
    In endoscopy the light source is co-located with the camera. Illumination
    falls off with 1/distance² — this is a property of scene geometry, not
    pixel intensity. A luminance-cluster mask coincidentally captures this
    sometimes but not reliably, which is why current outputs look arbitrary.

Depth convention (set in depth_estimator.py):
    depth ∈ [0, 1], 1.0 = nearest (camera), 0.0 = farthest (cavity).

Masks produced:
    overexposed:  mask_over  = depth ** γ_over          (peak at nearest)
    underexposed: mask_under = (1 - depth) ** γ_under   (peak at farthest)

Both are blended with a small fraction of the original cluster-based mask
so that texture-locked highlights / shadows still influence the result.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from exposure_augment import (
    rgb_to_lab, lab_to_rgb,
    decompose_luminance,
    find_seed_clusters,
    build_exposure_map,
)


def build_depth_mask(
    depth: np.ndarray,
    mode: str,
    gamma: float,
    smooth_sigma: float = 6.0,
) -> np.ndarray:
    """Compute a [0, 1] exposure influence map from depth.

    Args:
        depth: (H, W) float32 in [0, 1], 1.0 = near
        mode:  'overexposed' or 'underexposed'
        gamma: exponent concentrating effect (larger = more peaked on extremes)
        smooth_sigma: Gaussian smoothing on the mask (avoids sharp depth edges)

    Returns:
        (H, W) float32 in [0, 1]
    """
    d = depth.astype(np.float32)
    if mode == "overexposed":
        m = np.power(d, gamma)                 # peak at near
    else:
        m = np.power(1.0 - d, gamma)           # peak at far
    if smooth_sigma > 0:
        m = gaussian_filter(m, sigma=smooth_sigma)
    # renormalise so max = 1 (losing max to smoothing is ok, but we want
    # the peak region at full strength)
    mx = float(m.max())
    if mx > 1e-6:
        m = m / mx
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def depth_aware_augment(
    img_rgb: np.ndarray,
    depth: np.ndarray,
    mode: str = "underexposed",
    strength: float = 1.0,
    shift_magnitude: float = 70.0,
    # depth-mask controls
    gamma_over: float = 1.5,
    gamma_under: float = 2.0,
    depth_blend: float = 1.0,         # depth mask weight; 1-depth_blend = cluster mask
    smooth_sigma_base: float = 6.0,   # gaussian smoothing of depth mask (at 512px ref)
    # carry-over cluster controls (used for the minority blend)
    bright_percentile: float = 85.0,
    dark_percentile: float = 15.0,
    min_cluster_px: int = 50,
    texture_sigma: float = 3.0,
    # clip targets
    clip_target_over: float = 100.0,
    clip_target_under: float = 0.0,
    core_threshold: float = 0.65,     # influence above which we force clip
    core_clip_blend: float = 0.95,    # 1.0 = pure clip target in core, 0.0 = no extra pull
    hf_kill: float = 0.95,            # 1.0 = full HF removal in deepest core
) -> np.ndarray:
    """Depth-aware single-image exposure augmentation.

    Args:
        img_rgb: uint8 (H, W, 3) RGB input.
        depth:   float32 (H, W) in [0, 1], 1.0 = nearest.
        mode:    'overexposed' or 'underexposed'.
        strength: overall effect strength in [0, 1].
        shift_magnitude: maximum L-channel shift in absolute units [0, 100].

    Returns:
        uint8 (H, W, 3) RGB with exposure applied.
    """
    H, W = img_rgb.shape[:2]
    assert depth.shape == (H, W), f"depth {depth.shape} vs image {(H, W)}"

    # scale spatial constants relative to 512 px reference
    scale = max(H, W) / 512.0
    sigma_tex = texture_sigma * scale
    sigma_depth = smooth_sigma_base * scale
    min_cluster_scaled = int(min_cluster_px * scale * scale)

    # ── 1. LAB decomposition ──────────────────────────────────────────────
    lab = rgb_to_lab(img_rgb)
    L = lab[..., 0].copy()
    A = lab[..., 1].copy()
    B_ch = lab[..., 2].copy()
    L_low, L_high = decompose_luminance(L, texture_sigma=sigma_tex)

    # ── 2. Depth-driven exposure mask ─────────────────────────────────────
    gamma = gamma_over if mode == "overexposed" else gamma_under
    depth_mask = build_depth_mask(depth, mode, gamma=gamma, smooth_sigma=sigma_depth)

    # ── 3. Minor cluster contribution for texture-locked variation ────────
    if depth_blend < 1.0:
        seed_mask = find_seed_clusters(
            L, mode,
            bright_threshold_percentile=bright_percentile,
            dark_threshold_percentile=dark_percentile,
            min_cluster_pixels=min_cluster_scaled,
        )
        cluster_mask = build_exposure_map(
            seed_mask, L, mode,
            expand_radius_px=int(60 * scale),
            falloff_sigma=30.0 * scale,
            core_strength=1.0,
        )
    else:
        cluster_mask = np.zeros_like(depth_mask)

    exposure_map = depth_blend * depth_mask + (1.0 - depth_blend) * cluster_mask
    exposure_map = np.clip(exposure_map * strength, 0.0, 1.0).astype(np.float32)

    # ── 4. Apply luminance shift to the low-frequency envelope ────────────
    if mode == "overexposed":
        # boost brighter pixels harder (mimics saturation curve)
        brightness_boost = np.power(L_low / 100.0, 0.5)
        shift = exposure_map * shift_magnitude * (0.5 + 0.5 * brightness_boost)
        L_low_shifted = L_low + shift
        # hard clip at core — push aggressively toward clip_target_over
        core = exposure_map > core_threshold
        if core.any():
            L_low_shifted[core] = (
                (1.0 - core_clip_blend) * L_low_shifted[core]
                + core_clip_blend * clip_target_over
            )
    else:  # underexposed
        # pull darker pixels harder — and crush absolute darkness in core
        darkness_pull = np.power(1.0 - L_low / 100.0, 0.5)
        shift = exposure_map * shift_magnitude * (0.5 + 0.5 * darkness_pull)
        L_low_shifted = L_low - shift
        core = exposure_map > core_threshold
        if core.any():
            # Force hard black in the very deepest regions
            L_low_shifted[core] = (
                (1.0 - core_clip_blend) * L_low_shifted[core]
                + core_clip_blend * clip_target_under
            )

    # ── 5. Recombine with high-frequency texture ──────────────────────────
    # In core regions the HF detail would re-introduce mid-tone wiggle that
    # the eye reads as "dark textured" rather than truly black (or "bright
    # textured" rather than blown-out white). hf_kill controls how much of
    # the HF is removed at peak influence — at hf_kill=0.95 the deepest
    # core retains only 5% of the original texture amplitude.
    hf_attenuation = np.clip(1.0 - hf_kill * exposure_map, 0.0, 1.0)
    L_new = L_low_shifted + hf_attenuation * L_high

    L_new = np.clip(L_new, 0.0, 100.0).astype(np.float32)

    # ── 6. Reconstruct (AB preserved; chroma fix applied at inference) ────
    lab_new = np.stack([L_new, A, B_ch], axis=-1)
    return lab_to_rgb(lab_new)


def depth_aware_augment_lab(
    img_rgb: np.ndarray,
    depth: np.ndarray,
    mode: str = "underexposed",
    **kwargs,
) -> np.ndarray:
    """Variant that returns the full LAB array (H, W, 3) instead of RGB.

    Useful for saving paired training targets without double round-trip.
    """
    # Temporarily intercept the last step — replicate core pipeline then skip lab_to_rgb.
    # Simplest: call the RGB pipeline, convert back. Small overhead.
    rgb = depth_aware_augment(img_rgb, depth, mode=mode, **kwargs)
    return rgb_to_lab(rgb)

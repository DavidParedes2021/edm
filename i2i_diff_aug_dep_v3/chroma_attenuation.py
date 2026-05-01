#!/usr/bin/env python3
"""
chroma_attenuation.py — Physically-motivated AB-channel attenuation.

The single biggest perceptual defect in luminance-only exposure augmentation:
dropping L while keeping A,B unchanged produces DARK BROWN (for warm tissue)
instead of DARK BLACK. This module implements the fix.

Physical motivation (three stacking effects, all desaturate dark regions):
  1. Rod vision is achromatic — human color perception collapses in dim light
     (Purkinje effect). Dim scenes *should* read as near-monochrome.
  2. Image-sensor shot-noise floor — per-channel SNR collapses in low light,
     after which any ISP-level denoising effectively removes chroma.
  3. Specular highlight clipping — in overexposed regions, chroma also
     collapses because every channel clips to its maximum.

Output: (A', B') pair that replaces the original (A, B) before LAB→RGB.
No gradient descent, no model; deterministic, ~1 ms for a 1080p image.

Usage:
    from chroma_attenuation import attenuate_chroma
    A_new, B_new = attenuate_chroma(
        A, B,
        L_new=L_final, L_orig=L_orig,
        depth=depth,
        mode="underexposed",
        cfg=cfg["chroma"],
    )
"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np


DEFAULT_CFG = {
    "sat_knee": 5.0,              # L at which chroma multiplier hits sat_floor
    "sat_full": 40.0,             # L at which chroma multiplier hits 1.0
    "sat_floor": 0.10,            # minimum chroma multiplier (never exactly 0)
    "depth_chroma_boost": 0.30,   # extra desat for far regions (0 = disabled)
    "highlight_desat": 0.60,      # chroma cut when L > 90 (overexposed clipping)
    "highlight_knee": 90.0,       # L above which highlight_desat kicks in
    "purkinje_b_shift": 1.5,      # max negative B shift in very dark pixels
    "purkinje_threshold": 20.0,   # below this L, Purkinje shift ramps up
}


def _smooth_ramp(x: np.ndarray, lo: float, hi: float,
                 out_lo: float = 0.0, out_hi: float = 1.0) -> np.ndarray:
    """Smoothstep from out_lo at x<=lo to out_hi at x>=hi."""
    t = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    # cubic smoothstep (C¹ continuous)
    t = t * t * (3.0 - 2.0 * t)
    return out_lo + (out_hi - out_lo) * t


def attenuate_chroma(
    A: np.ndarray,
    B: np.ndarray,
    L_new: np.ndarray,
    L_orig: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    mode: str = "underexposed",
    cfg: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A', B') after physically-motivated chroma attenuation.

    Args:
        A, B:     (H, W) float32 LAB chroma channels, A/B ∈ [-128, 127]
        L_new:    (H, W) float32 new luminance, [0, 100]
        L_orig:   (H, W) float32 original luminance (optional, for ΔL modulation)
        depth:    (H, W) float32 in [0, 1], 1.0 = nearest, 0.0 = farthest
        mode:     'overexposed' or 'underexposed'
        cfg:      dict of constants (see DEFAULT_CFG). Missing keys take defaults.

    Returns:
        (A_new, B_new) float32 arrays of the same shape.
    """
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)

    # ── Base saturation ramp driven by absolute luminance ─────────────────
    # s = sat_floor at L<=knee, 1.0 at L>=full, smooth in between.
    sat = _smooth_ramp(L_new, c["sat_knee"], c["sat_full"],
                       out_lo=c["sat_floor"], out_hi=1.0)

    # ── Depth-modulated extra desaturation in far regions ─────────────────
    # Far = low depth, where illumination would fall off and noise dominate.
    if depth is not None and c["depth_chroma_boost"] > 0:
        # (1 - depth) is "farness" in [0, 1]
        far = 1.0 - depth.astype(np.float32)
        sat = sat * (1.0 - c["depth_chroma_boost"] * far)

    # ── Overexposed highlight desaturation ────────────────────────────────
    # Clipped highlights desaturate in real cameras because all channels
    # saturate together. Apply a smooth cut for L above highlight_knee.
    if mode == "overexposed" and c["highlight_desat"] > 0:
        over = _smooth_ramp(L_new, c["highlight_knee"], 100.0, 0.0, 1.0)
        sat = sat * (1.0 - c["highlight_desat"] * over)

    # Clamp to a safe range (never exactly zero → avoids abrupt hue banding)
    sat = np.clip(sat, c["sat_floor"] * 0.8, 1.0).astype(np.float32)

    # ── Apply saturation scale ────────────────────────────────────────────
    A_new = (A.astype(np.float32) * sat)
    B_new = (B.astype(np.float32) * sat)

    # ── Subtle cool (Purkinje) shift in very dark pixels ──────────────────
    # Shifts B negative (toward blue) as L drops below `purkinje_threshold`.
    if mode == "underexposed" and c["purkinje_b_shift"] > 0:
        purk = _smooth_ramp(
            L_new, 0.0, c["purkinje_threshold"], 1.0, 0.0  # 1 at L=0, 0 at L=thresh
        )
        B_new = B_new - c["purkinje_b_shift"] * purk

    # Clip to valid LAB chroma range
    A_new = np.clip(A_new, -128.0, 127.0).astype(np.float32)
    B_new = np.clip(B_new, -128.0, 127.0).astype(np.float32)

    return A_new, B_new


# ──────────────────────────────────────────────────────────────────────────
# Texture gate — attenuate HF detail in extreme L regions
# ──────────────────────────────────────────────────────────────────────────
def texture_gate(L_low_pred: np.ndarray, mode: str,
                 knee: float = 30.0, floor: float = 0.05) -> np.ndarray:
    """Smooth gate that attenuates high-frequency detail toward a `floor`
    in extreme regions of the predicted low-frequency luminance.

    Without this, recombining `L_low_pred + L_high_orig` re-injects the
    original mid-tone texture variation into pixels the model wanted to
    drive toward black or white — the dominant cause of "barely noticeable"
    underexposure even when the predicted L_low is correct.

        underexposed: gate = floor at L_low_pred=0, gate = 1 at L_low_pred>=knee
        overexposed:  gate = 1 at L_low_pred<=100-knee, gate = floor at L_low_pred=100

    Args:
        L_low_pred: (H, W) float32 in [0, 100], the model's predicted low-pass L.
        mode: 'overexposed' or 'underexposed'.
        knee: width of the smooth ramp in L units.
        floor: minimum multiplier in the deepest extreme (>0 to avoid banding).

    Returns:
        (H, W) float32 in [floor, 1.0].
    """
    if mode == "underexposed":
        return _smooth_ramp(L_low_pred, 0.0, knee, out_lo=floor, out_hi=1.0)
    elif mode == "overexposed":
        return _smooth_ramp(L_low_pred, 100.0 - knee, 100.0, out_lo=1.0, out_hi=floor)
    else:
        raise ValueError(f"mode must be 'overexposed' or 'underexposed', got {mode}")


# ──────────────────────────────────────────────────────────────────────────
# Convenience: full LAB → LAB chroma-attenuated
# ──────────────────────────────────────────────────────────────────────────
def attenuate_lab(
    lab_new: np.ndarray,
    lab_orig: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    mode: str = "underexposed",
    cfg: Optional[dict] = None,
) -> np.ndarray:
    """Wrap `attenuate_chroma` to take/return a full (H,W,3) LAB array."""
    L = lab_new[..., 0]
    A = lab_new[..., 1]
    B = lab_new[..., 2]
    L_orig = lab_orig[..., 0] if lab_orig is not None else None
    A_new, B_new = attenuate_chroma(
        A, B, L_new=L, L_orig=L_orig, depth=depth, mode=mode, cfg=cfg
    )
    return np.stack([L, A_new, B_new], axis=-1)


if __name__ == "__main__":
    # quick sanity check
    import matplotlib.pyplot as plt  # optional

    L_sweep = np.linspace(0, 100, 101, dtype=np.float32)
    A = np.full_like(L_sweep, 20.0)
    B = np.full_like(L_sweep, 25.0)
    A_u, B_u = attenuate_chroma(A, B, L_new=L_sweep, mode="underexposed")
    A_o, B_o = attenuate_chroma(A, B, L_new=L_sweep, mode="overexposed")

    print(f"{'L':>6} {'A_under':>8} {'B_under':>8} {'A_over':>8} {'B_over':>8}")
    for i in (0, 10, 20, 40, 60, 80, 100):
        print(f"{L_sweep[i]:6.1f} {A_u[i]:8.2f} {B_u[i]:8.2f} "
              f"{A_o[i]:8.2f} {B_o[i]:8.2f}")

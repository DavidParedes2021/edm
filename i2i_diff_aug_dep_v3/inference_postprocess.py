"""
inference_postprocess.py — Robustness helpers for inference.

The diffusion model is trained on whole frames, but real endoscopy frames
contain large non-content regions:
  • UI panels with text overlays (uniform black background)
  • Letterboxing / circular FOV mask outside the round endoscope window

On those pixels the model has no useful conditioning (source_L ≈ −1, depth
is whatever MiDaS hallucinated on uniform black). DDIM cannot fully denoise
without signal, so the output L there is noisy. The depth estimator's
per-image min-max also gets skewed by those black pixels, distorting the
[0, 1] mapping of the actual scene.

This module provides three post-process helpers used at inference and during
training-time sample previews:

  detect_content_mask(rgb)         → boolean (H, W) of valid scene area
  clean_depth_with_mask(d, mask)   → renormalised depth, borders → "near" (1.0)
  focal_blend_alpha(d, mode, gamma)→ depth-driven blend weight α(d)

The combined recipe at the call site:

  1. mask = detect_content_mask(rgb_orig)
  2. d_clean = clean_depth_with_mask(d_full, mask)         # for chroma + blend
  3. (model is run on a model-resolution version of d_clean)
  4. L_final = focal_blend_alpha(d_clean, mode) * L_pred
             + (1 - focal_blend_alpha(d_clean, mode)) * L_orig
  5. L_final[~mask] = L_orig[~mask]
  6. AB chroma also passthrough outside the mask
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    label as connected_components,
)


def detect_content_mask(
    rgb: np.ndarray,
    luma_threshold: float = 8.0,
    min_content_fraction: float = 0.05,
    erode_iter: int = 2,
) -> np.ndarray:
    """Detect the valid endoscopic scene area.

    Strategy: pixels with luma > threshold are content candidates. A
    morphological opening removes specks, the largest connected blob is
    kept, holes are filled (so that bright-on-black UI text inside a black
    panel is *not* re-classified as content), and the boundary is eroded
    a few pixels to keep the effect from leaking onto the FOV edge.

    Args:
        rgb: uint8 (H, W, 3) RGB.
        luma_threshold: Rec.601 luma below which a pixel is "black-ish".
        min_content_fraction: safety. If detection collapses (e.g. the frame
            is entirely dark) we accept the whole image rather than mask all.
        erode_iter: erosion iterations on the final mask.

    Returns:
        bool (H, W) — True = content, False = UI/border.
    """
    luma = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    mask = luma > luma_threshold

    # Strip text-on-black artefacts: opening removes thin overlay strokes
    mask = binary_opening(mask, iterations=2)

    # Keep only the largest connected component (the round FOV)
    lab, n = connected_components(mask)
    if n > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0  # background
        biggest = int(np.argmax(sizes))
        mask = (lab == biggest)

    # Fill holes so vessels / dark crypts inside the FOV count as content
    mask = binary_fill_holes(mask)

    if mask.mean() < min_content_fraction:
        # detection collapsed — fail open
        return np.ones(luma.shape, dtype=bool)

    if erode_iter > 0:
        mask = binary_erosion(mask, iterations=erode_iter)

    return mask.astype(bool)


def clean_depth_with_mask(
    depth: np.ndarray,
    content_mask: np.ndarray,
    border_value: float = 1.0,
) -> np.ndarray:
    """Renormalise depth within the content mask and force borders to a
    benign value.

    The cached depth from `depth_estimator.py` was min-max normalised over
    the whole frame including UI panels. UI pixels can pin either end of
    the [0, 1] range, compressing the actual scene's depth into a sub-range.
    Recompute the min-max using only content pixels, then assign `border_value`
    to non-content pixels.

    Args:
        depth: float32 (H, W) in roughly [0, 1].
        content_mask: bool (H, W).
        border_value: depth assigned to non-content. 1.0 = "near", which
            yields zero underexposure mask (and zero focal blend weight).

    Returns:
        float32 (H, W) in [0, 1].
    """
    d = depth.astype(np.float32, copy=True)
    if content_mask.any():
        vals = d[content_mask]
        mn, mx = float(vals.min()), float(vals.max())
        if mx - mn > 1e-6:
            d = (d - mn) / (mx - mn)
        # else: leave d as-is (degenerate, content is flat)
    d = np.clip(d, 0.0, 1.0)
    d[~content_mask] = border_value
    return d


def focal_blend_alpha(
    depth: np.ndarray,
    mode: str,
    gamma: float = 2.5,
    knee: float = 0.05,
    floor: float = 0.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Depth-driven blend weight: α(d) ∈ [floor, 1].

    For 'underexposed', α peaks at depth=0 (far/cavity) and falls to `floor`
    at depth=1 (near/light source). For 'overexposed', it is mirrored.

    A small `knee` flattens the bottom — pixels above `knee` of "depth-distance
    from extreme" get α=floor (no model effect). This is what makes the
    output focal: the model's prediction is only allowed to act where the
    depth says it should.

    Args:
        depth: float32 (H, W) in [0, 1] (1 = near, 0 = far).
        mode: 'underexposed' or 'overexposed'.
        gamma: exponent. Larger = more concentrated peak.
        knee: dead-zone at the wrong end (default 0.05).
        floor: minimum α (default 0.0 — fully suppress effect at near tissue).
        smooth_sigma: optional Gaussian smoothing on α (in pixels).

    Returns:
        float32 (H, W) in [0, 1].
    """
    d = np.clip(depth.astype(np.float32), 0.0, 1.0)
    if mode == "underexposed":
        # peak at d=0 (far) — "farness"
        x = 1.0 - d
    elif mode == "overexposed":
        # peak at d=1 (near) — "nearness"
        x = d
    else:
        raise ValueError(f"mode must be 'underexposed' or 'overexposed', got {mode}")

    # Apply knee: only the (1-knee) top of the range counts
    x = np.clip((x - knee) / (1.0 - knee + 1e-8), 0.0, 1.0)
    a = np.power(x, gamma)
    a = floor + (1.0 - floor) * a

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, sigma=smooth_sigma)
        a = np.clip(a, 0.0, 1.0)

    return a.astype(np.float32)

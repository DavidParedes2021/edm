"""Depth pseudo-estimation and vulnerability masks for focalization.

Why a luminance-based pseudo-depth?
-----------------------------------
In endoscopy the light source is co-located with the camera, so radiance falls
off rapidly with distance: dark regions are far (deep lumen), bright regions
are close (mucosa wall). This gives a robust, offline, dependency-free depth
proxy that matches the user's stated heuristic perfectly. A real metric depth
model (e.g. MiDaS) can be plugged in via ``depth_from_rgb`` -> swap the
implementation; the mask logic does not change.

Vulnerability masks
-------------------
For OVEREXPOSURE: the regions that *naturally* become saturated are those that
are already bright AND close to the endoscope. mask_over ∝ Y * (1 - depth).

For UNDEREXPOSURE: the regions that *naturally* become invisible are those
that are already dark AND far away. mask_under ∝ (1 - Y) * depth.

The masks are smoothed and percentile-normalized so they describe focal
*clusters*, not noise. They are used at inference both:
    (a) as a hard RePaint constraint at every diffusion step, and
    (b) as a final blend factor: y = mask * y_gen + (1 - mask) * y_orig.
This is what makes the change focalized rather than dispersed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    half = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. x: (B, C, H, W). Reflect padding."""
    if sigma <= 0:
        return x
    g = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    k = g.numel()
    pad = k // 2
    C = x.shape[1]
    kh = g.view(1, 1, 1, k).expand(C, 1, 1, k)
    kv = g.view(1, 1, k, 1).expand(C, 1, k, 1)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    x = F.conv2d(x, kv, groups=C)
    return x


def _percentile(t: torch.Tensor, q: float) -> torch.Tensor:
    """q in [0, 100]. Returns scalar tensor (the q-th percentile of all elements)."""
    flat = t.flatten()
    n = flat.numel()
    k = max(1, min(n, int(round(q / 100.0 * n))))
    return flat.kthvalue(k).values


def percentile_normalize(x: torch.Tensor, p_low: float = 1.0, p_high: float = 99.0) -> torch.Tensor:
    """Per-image robust min-max -> [0, 1] using the given percentiles."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        lo = _percentile(x[i], p_low)
        hi = _percentile(x[i], p_high)
        out[i] = ((x[i] - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Depth pseudo-estimation
# ---------------------------------------------------------------------------

def luminance_pseudo_depth(Y_norm: torch.Tensor, blur_sigma: float = 8.0) -> torch.Tensor:
    """Depth proxy from luminance.

    Y_norm: (B, 1, H, W) in [-1, 1].
    Returns: (B, 1, H, W) in [0, 1] where 0 = close, 1 = far.
    """
    Y01 = ((Y_norm + 1.0) / 2.0).clamp(0.0, 1.0)
    raw = 1.0 - Y01
    smooth = gaussian_blur_2d(raw, blur_sigma)
    return percentile_normalize(smooth)


def depth_from_rgb(rgb: torch.Tensor, blur_sigma: float = 8.0) -> torch.Tensor:
    """Convenience wrapper: compute depth pseudo-estimate from RGB directly.

    rgb: (B, 3, H, W) in [0, 1]. Returns (B, 1, H, W) in [0, 1].
    """
    Y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    Y_norm = Y * 2.0 - 1.0
    return luminance_pseudo_depth(Y_norm, blur_sigma)


# ---------------------------------------------------------------------------
# Vulnerability masks
# ---------------------------------------------------------------------------

def vulnerability_masks(
    Y_norm: torch.Tensor,
    depth: torch.Tensor,
    blur_sigma: float = 16.0,
    gamma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute focal-cluster masks for over- and under-exposure.

    Args
    ----
    Y_norm : (B, 1, H, W) in [-1, 1] -- the luminance of the input frame.
    depth  : (B, 1, H, W) in  [0, 1] -- 0 = close, 1 = far (from depth_from_rgb).
    blur_sigma : Gaussian sigma applied to the raw masks (controls cluster size).
    gamma : exponent applied after percentile-normalization (>1 sharpens clusters).

    Returns
    -------
    (mask_over, mask_under), each (B, 1, H, W) in [0, 1].
    """
    Y01 = ((Y_norm + 1.0) / 2.0).clamp(0.0, 1.0)
    closeness = 1.0 - depth
    darkness = 1.0 - Y01

    raw_over = Y01 * closeness            # bright + close = focal hot spot
    raw_under = darkness * depth          # dark  + far   = focal shadow

    raw_over = gaussian_blur_2d(raw_over, blur_sigma)
    raw_under = gaussian_blur_2d(raw_under, blur_sigma)

    m_over = percentile_normalize(raw_over).pow(gamma)
    m_under = percentile_normalize(raw_under).pow(gamma)

    return m_over, m_under

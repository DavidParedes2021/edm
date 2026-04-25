"""LAB <-> RGB conversion helpers.

We diffuse only on the CIE-LAB L* channel and substitute it back into the
original (a*, b*) without modification, guaranteeing zero chrominance drift.
"""

import warnings

import numpy as np
from skimage import color as _skcolor

# Diffused L* combined with the original (a*, b*) can land slightly outside
# the sRGB gamut; skimage logs a UserWarning and clips. The clip is exactly
# what we want, so silence this benign warning.
warnings.filterwarnings(
    "ignore",
    message=r"Conversion from CIE-LAB.*",
    category=UserWarning,
)


def rgb_to_lab(img_rgb_uint8: np.ndarray):
    """img_rgb_uint8: (H, W, 3) uint8 RGB.

    Returns:
        L:  (H, W) float32 in [0, 100]
        ab: (H, W, 2) float32 (skimage scale, ~[-128, 127])
    """
    if img_rgb_uint8.dtype != np.uint8:
        img_rgb_uint8 = img_rgb_uint8.astype(np.uint8)
    lab = _skcolor.rgb2lab(img_rgb_uint8)
    L = lab[..., 0].astype(np.float32)
    ab = lab[..., 1:].astype(np.float32)
    return L, ab


def lab_to_rgb(L_0_100: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """L: (H, W) [0, 100], ab: (H, W, 2). Returns (H, W, 3) uint8 RGB."""
    lab = np.concatenate([L_0_100[..., None], ab], axis=-1).astype(np.float64)
    rgb = _skcolor.lab2rgb(lab)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def normalize_L(L_0_100: np.ndarray) -> np.ndarray:
    """Map [0, 100] -> [-1, 1]."""
    return (L_0_100 / 50.0) - 1.0


def denormalize_L(L_pm1: np.ndarray) -> np.ndarray:
    """Map [-1, 1] -> [0, 100]."""
    return (L_pm1 + 1.0) * 50.0

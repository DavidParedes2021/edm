"""LAB <-> RGB conversions and luminance recombination.

Re-exposes the pure-NumPy LAB conversions from `code _to_generate_pairs/exposure_augment.py`
so this package is importable without touching the original generation code,
while remaining bit-identical to the rule-based augmenter the targets were
produced with.

Recombination rule (the design choice the user asked for):
    - Take A,B from the original normal RGB.
    - Replace L with the diffusion-predicted target L.
    - Convert (L_pred, A, B) → RGB.
    Texture and colour are preserved; only luminance is impacted.

If `use_residual=True` the predicted residual (target_L_pred - normal_L_lowres)
can be upsampled to the full image resolution and added to the *full-resolution*
normal_L. That is handled in `inference.py`; this module only handles the
final LAB→RGB step and array I/O.
"""
from __future__ import annotations

import numpy as np


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB → float32 HWC LAB."""
    x = img.astype(np.float32) / 255.0
    mask = x > 0.04045
    x = np.where(mask, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
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
    mat_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    rgb = xyz @ mat_inv.T
    rgb = np.clip(rgb, 0.0, None)
    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055, 12.92 * rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def recombine_L_with_chroma(L_target: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stack predicted L with original A,B chroma and convert to RGB."""
    if not (L_target.shape == A.shape == B.shape):
        raise ValueError(
            f"Shape mismatch: L={L_target.shape} A={A.shape} B={B.shape}"
        )
    L_target = np.clip(L_target.astype(np.float32), 0.0, 100.0)
    lab = np.stack([L_target, A.astype(np.float32), B.astype(np.float32)], axis=-1)
    return lab_to_rgb(lab)

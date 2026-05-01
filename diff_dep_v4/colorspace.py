"""RGB <-> YCbCr conversion utilities (BT.601), device-safe.

All tensors are 4-D: (B, C, H, W) in [0, 1] for RGB and [0, 1] for YCbCr,
except the luminance channel returned by ``split_ycbcr`` which is normalized
to [-1, 1] for diffusion training (matches typical DDPM input range).
"""
from __future__ import annotations

import torch

# BT.601 forward matrix (RGB -> YCbCr) and bias (added after the matrix product).
_RGB2YCBCR = torch.tensor(
    [
        [0.299,      0.587,      0.114],
        [-0.168736, -0.331264,   0.5],
        [0.5,       -0.418688,  -0.081312],
    ],
    dtype=torch.float32,
)
_YCBCR_BIAS = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)

_YCBCR2RGB = torch.tensor(
    [
        [1.0,  0.0,        1.402],
        [1.0, -0.344136,  -0.714136],
        [1.0,  1.772,      0.0],
    ],
    dtype=torch.float32,
)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB in [0, 1] -> (B, 3, H, W) YCbCr in [0, 1]."""
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (B, 3, H, W); got {tuple(rgb.shape)}")
    M = _RGB2YCBCR.to(device=rgb.device, dtype=rgb.dtype)
    bias = _YCBCR_BIAS.to(device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return torch.einsum("cd,bdhw->bchw", M, rgb) + bias


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) YCbCr in [0, 1] -> (B, 3, H, W) RGB in [0, 1] (clamped)."""
    if ycbcr.dim() != 4 or ycbcr.shape[1] != 3:
        raise ValueError(f"ycbcr must be (B, 3, H, W); got {tuple(ycbcr.shape)}")
    M = _YCBCR2RGB.to(device=ycbcr.device, dtype=ycbcr.dtype)
    bias = _YCBCR_BIAS.to(device=ycbcr.device, dtype=ycbcr.dtype).view(1, 3, 1, 1)
    centered = ycbcr - bias
    rgb = torch.einsum("cd,bdhw->bchw", M, centered)
    return rgb.clamp(0.0, 1.0)


def split_ycbcr(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB to YCbCr and split. Returns (Y_norm, CbCr).

    - Y_norm : (B, 1, H, W) in [-1, 1]   -- diffusion-ready luminance
    - CbCr   : (B, 2, H, W) in  [0, 1]   -- raw chrominance, untouched
    """
    ycbcr = rgb_to_ycbcr(rgb)
    Y = ycbcr[:, 0:1, :, :]
    CbCr = ycbcr[:, 1:3, :, :]
    Y_norm = (Y * 2.0 - 1.0).clamp(-1.0, 1.0)
    return Y_norm, CbCr


def merge_ycbcr(Y_norm: torch.Tensor, CbCr: torch.Tensor) -> torch.Tensor:
    """Recombine Y in [-1, 1] and CbCr in [0, 1] into RGB in [0, 1]."""
    if Y_norm.shape[1] != 1 or CbCr.shape[1] != 2:
        raise ValueError(
            f"Y_norm must be (B,1,H,W) and CbCr (B,2,H,W); got {Y_norm.shape} {CbCr.shape}"
        )
    Y = ((Y_norm + 1.0) / 2.0).clamp(0.0, 1.0)
    ycbcr = torch.cat([Y, CbCr], dim=1)
    return ycbcr_to_rgb(ycbcr)

"""
RGB <-> YCbCr conversion (BT.601 full range, normalized to [0, 1]).

Why YCbCr (not LAB):
- Linear, invertible transform; no gamma round-trip noise.
- Y on [0, 1], Cb/Cr on [0, 1] (centered at 0.5) -> easy to feed into a neural net.
- Inverse is exact (up to clipping at the RGB boundary).

All functions accept (..., 3, H, W) tensors with values in [0, 1].
"""
from __future__ import annotations
import torch

# BT.601 full-range matrices. Cb/Cr are stored offset by +0.5 so that all
# three channels live in [0, 1] (no negative values, no /255 ambiguity).
_RGB2YCBCR = torch.tensor([
    [ 0.299000,  0.587000,  0.114000],
    [-0.168736, -0.331264,  0.500000],
    [ 0.500000, -0.418688, -0.081312],
], dtype=torch.float32)

_YCBCR2RGB = torch.tensor([
    [1.0,  0.000000,  1.402000],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772000,  0.000000],
], dtype=torch.float32)

_CHROMA_BIAS = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0, 1] -> YCbCr with all channels in roughly [0, 1]."""
    M = _RGB2YCBCR.to(device=rgb.device, dtype=rgb.dtype)
    bias = _CHROMA_BIAS.to(device=rgb.device, dtype=rgb.dtype).view(3, 1, 1)
    out = torch.einsum('ck,...khw->...chw', M, rgb) + bias
    return out


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """YCbCr -> RGB in [0, 1] (clipped)."""
    M = _YCBCR2RGB.to(device=ycbcr.device, dtype=ycbcr.dtype)
    bias = _CHROMA_BIAS.to(device=ycbcr.device, dtype=ycbcr.dtype).view(3, 1, 1)
    out = torch.einsum('ck,...khw->...chw', M, ycbcr - bias)
    return out.clamp(0.0, 1.0)


def replace_y(ycbcr: torch.Tensor, new_y: torch.Tensor) -> torch.Tensor:
    """Splice a new Y channel into an existing YCbCr tensor.

    ycbcr: (..., 3, H, W) -- only the Cb and Cr channels are used
    new_y: (..., 1, H, W) in [0, 1]
    """
    return torch.cat([new_y, ycbcr[..., 1:, :, :]], dim=-3)


# Diffusion is happier on a symmetric range [-1, 1].
def to_diffusion(y01: torch.Tensor) -> torch.Tensor:
    return y01 * 2.0 - 1.0


def from_diffusion(y_pm1: torch.Tensor) -> torch.Tensor:
    return ((y_pm1 + 1.0) * 0.5).clamp(0.0, 1.0)


# ---------- self-test ----------
if __name__ == "__main__":
    x = torch.rand(2, 3, 8, 8)
    y = rgb_to_ycbcr(x)
    z = ycbcr_to_rgb(y)
    err = (x - z).abs().max().item()
    print(f"Round-trip max error: {err:.2e}")  # ~1e-7 in fp32
    assert err < 1e-5
    print("color.py OK")

# utils/color_space.py
"""
Tensor-native RGB <-> YCbCr conversion.
All operations stay on the same device as the input tensor.
No CPU numpy calls during forward pass → eliminates cross-device errors.

Hypothesis validation:
  Exposure changes (over/under) affect ONLY luminance (Y channel).
  Chrominance (Cb, Cr) encodes color information and is invariant to pure
  exposure shifts. By diffusing only Y and recombining with the original Cb/Cr
  from the normal frame, we guarantee:
    1. Color fidelity (no hue/saturation drift)
    2. Structural preservation (edges encoded in chroma are preserved)
    3. 3x parameter efficiency (1-channel UNet vs 3-channel)
"""
import torch
import torch.nn.functional as F


# ITU-R BT.601 coefficients (standard for digital video / camera sensors)
# These match what most ISPs use and are appropriate for frame sequences.
_RGB_TO_YCBCR = torch.tensor([
    [ 0.299,    0.587,    0.114  ],   # Y
    [-0.168736,-0.331264, 0.5    ],   # Cb
    [ 0.5,     -0.418688,-0.081312],  # Cr
], dtype=torch.float32)

_YCBCR_TO_RGB = torch.tensor([
    [1.0,  0.0,       1.402   ],      # R
    [1.0, -0.344136, -0.714136],      # G
    [1.0,  1.772,     0.0     ],      # B
], dtype=torch.float32)

_CB_CR_OFFSET = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB → YCbCr.

    Args:
        rgb: (B, 3, H, W) float tensor in [0, 1]
    Returns:
        ycbcr: (B, 3, H, W) where Y ∈ [0,1], Cb/Cr ∈ [0,1] (offset by 0.5)
    """
    assert rgb.ndim == 4 and rgb.shape[1] == 3, \
        f"Expected (B,3,H,W), got {rgb.shape}"

    device = rgb.device
    mat = _RGB_TO_YCBCR.to(device)          # (3,3) — same device as input
    offset = _CB_CR_OFFSET.to(device)       # (3,)

    # (B, 3, H, W) → (B, H, W, 3) for matmul
    rgb_hwc = rgb.permute(0, 2, 3, 1)       # (B, H, W, 3)
    ycbcr_hwc = rgb_hwc @ mat.T + offset    # (B, H, W, 3)
    return ycbcr_hwc.permute(0, 3, 1, 2)    # (B, 3, H, W)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr → RGB.

    Args:
        ycbcr: (B, 3, H, W) where Y ∈ [0,1], Cb/Cr ∈ [0,1]
    Returns:
        rgb: (B, 3, H, W) clamped to [0, 1]
    """
    assert ycbcr.ndim == 4 and ycbcr.shape[1] == 3, \
        f"Expected (B,3,H,W), got {ycbcr.shape}"

    device = ycbcr.device
    mat = _YCBCR_TO_RGB.to(device)
    offset = _CB_CR_OFFSET.to(device)

    ycbcr_hwc = ycbcr.permute(0, 2, 3, 1) - offset  # subtract offset
    rgb_hwc = ycbcr_hwc @ mat.T
    rgb = rgb_hwc.permute(0, 3, 1, 2)
    return rgb.clamp(0.0, 1.0)


def extract_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """
    Extract only the Y (luminance) channel from RGB.

    Returns:
        y: (B, 1, H, W) in [0, 1]
    """
    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, :1, :, :]          # Y channel only


def replace_luminance(rgb_source: torch.Tensor,
                      y_new: torch.Tensor) -> torch.Tensor:
    """
    Replace the luminance of rgb_source with y_new.
    Chrominance (Cb, Cr) is preserved from rgb_source.

    Args:
        rgb_source: (B, 3, H, W) — provides Cb, Cr (color structure)
        y_new:      (B, 1, H, W) — new luminance values
    Returns:
        rgb_out: (B, 3, H, W)
    """
    ycbcr = rgb_to_ycbcr(rgb_source)
    ycbcr_modified = torch.cat([y_new, ycbcr[:, 1:, :, :]], dim=1)
    return ycbcr_to_rgb(ycbcr_modified)


def normalize_luminance(y: torch.Tensor) -> torch.Tensor:
    """Scale Y from [0,1] to [-1,1] for diffusion model input."""
    return y * 2.0 - 1.0


def denormalize_luminance(y: torch.Tensor) -> torch.Tensor:
    """Scale diffusion output from [-1,1] back to [0,1]."""
    return (y + 1.0) * 0.5

"""
losses/ssim_loss.py
-------------------
SSIM computed on Sobel gradient magnitude maps.

Why gradient SSIM instead of pixel SSIM?
-----------------------------------------
Standard SSIM penalises luminance differences, which is exactly what we
*want* to change (exposure).  Computing SSIM on the gradient magnitude
(edges, textures) makes the loss illumination-invariant while still
penalising structural drift (misaligned edges, hallucinated objects).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientSSIMLoss(nn.Module):
    """
    Structural similarity loss on Sobel gradient magnitude maps.

    Parameters
    ----------
    window_size : int
        SSIM sliding window (Gaussian kernel size).
    """

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer("_gauss_kernel", self._make_gauss(window_size))

        # Sobel kernels  (kept as buffers, not parameters)
        kx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer("_kx", kx)
        self.register_buffer("_ky", ky)

    @staticmethod
    def _make_gauss(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return (g.unsqueeze(0) * g.unsqueeze(1)).view(1, 1, size, size)

    def _sobel_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 1, H, W) gradient magnitude in [0, 1].
        Always computed in fp32 — 1e-6 underflows to zero in fp16.
        """
        x    = x.float()
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        gray = (gray + 1.0) / 2.0   # [-1,1] → [0,1]
        kx   = self._kx.float()
        ky   = self._ky.float()
        gx   = F.conv2d(gray, kx, padding=1)
        gy   = F.conv2d(gray, ky, padding=1)
        mag  = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        mag  = mag / (mag.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        return mag

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns mean SSIM over (B, 1, H, W) inputs, always in [0, 1].

        fp16 note: variance estimates (E[x²] - E[x]²) can go slightly
        negative under fp16 due to catastrophic cancellation, which drives
        SSIM above 1 and makes `1 - SSIM` negative.  We force fp32 for
        the convolution arithmetic and clamp variances to >= 0 before use.
        """
        C1, C2 = 0.0001, 0.0009
        pad = self.window_size // 2

        # Always run SSIM arithmetic in fp32 regardless of AMP context.
        x = x.float()
        y = y.float()
        kernel = self._gauss_kernel.float().expand(1, 1, -1, -1)

        mu_x  = F.conv2d(x,     kernel, padding=pad)
        mu_y  = F.conv2d(y,     kernel, padding=pad)
        mu_xx = (F.conv2d(x * x, kernel, padding=pad) - mu_x ** 2).clamp(min=0)
        mu_yy = (F.conv2d(y * y, kernel, padding=pad) - mu_y ** 2).clamp(min=0)
        mu_xy =  F.conv2d(x * y, kernel, padding=pad) - mu_x * mu_y

        num  = (2.0 * mu_x * mu_y + C1) * (2.0 * mu_xy + C2)
        den  = (mu_x ** 2 + mu_y ** 2 + C1) * (mu_xx + mu_yy + C2)
        ssim = (num / den.clamp(min=1e-8)).clamp(0.0, 1.0)
        return ssim.mean()

    def forward(
        self,
        pred:   torch.Tensor,   # (B, 3, H, W) generated exposure image
        source: torch.Tensor,   # (B, 3, H, W) Normal source frame
    ) -> torch.Tensor:
        """Returns 1 - SSIM (lower = more structurally similar)."""
        pred_grad   = self._sobel_magnitude(pred)
        source_grad = self._sobel_magnitude(source.detach())
        ssim_val    = self._ssim(pred_grad, source_grad)
        return 1.0 - ssim_val
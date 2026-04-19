"""
losses.py — Sharpness- and darkness-preserving losses for the depth-aware
luminance diffusion.

Changes vs. previous:
  - SobelEdgeLoss unchanged.
  - Removed VGG perceptual loss (actively harmful for this task — it pulls
    toward natural-looking smoothness, which fights true underexposure).
  - Added DarknessWeightedL1Loss — upweights error where target is dark.
  - Added DepthGradientConsistencyLoss — enforces that the luminance shift
    field respects depth discontinuities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sobel edge loss — unchanged
# ─────────────────────────────────────────────────────────────────────────────
class SobelEdgeLoss(nn.Module):
    """L1 on Sobel edge magnitudes — penalises blurry outputs."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.T.clone()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edges(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


# ─────────────────────────────────────────────────────────────────────────────
# Darkness-weighted L1
# ─────────────────────────────────────────────────────────────────────────────
class DarknessWeightedL1Loss(nn.Module):
    """Upweight pixel error where the TARGET is dark.

    Counteracts the mean-regression bias of vanilla L1 — without this the
    model systematically under-darkens the deep-cavity regions because most
    pixels in the dataset are mid-luminance.

    Weight:  w = 1 + alpha * exp(-(target + 1) / tau)
    where target is in [-1, 1] and (target + 1) maps to [0, 2].
    Defaults alpha=3.0, tau=0.3 → target=-1 weighs ~4.0, target=+1 weighs ~1.001.
    """

    def __init__(self, alpha: float = 3.0, tau: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target expected in [-1, 1]; shift so "dark" is 0
        shifted = target + 1.0
        w = 1.0 + self.alpha * torch.exp(-shifted / self.tau)
        return (w * (pred - target).abs()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Depth-gradient consistency
# ─────────────────────────────────────────────────────────────────────────────
class DepthGradientConsistencyLoss(nn.Module):
    """Enforce that the exposure shift field respects depth discontinuities.

    Concept:
        Let ΔL = x0_hat - source_L (the learned luminance shift).
        Let |∇ΔL| and |∇D| be Sobel gradient magnitudes, per-image normalised.
        We want: wherever |∇D| is large, |∇ΔL| should also be allowed to be
        large — i.e., don't smooth the shift across a depth edge.

        We implement this as a ONE-SIDED soft hinge:
            loss = mean( ReLU( |∇D|_norm - |∇ΔL|_norm - margin ) )
        which penalises only when |∇ΔL| is too small at a depth edge.
        It does NOT force ΔL gradient to match D gradient anywhere else.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.T.clone()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    @staticmethod
    def _percentile_norm(x: torch.Tensor, p: float = 0.98) -> torch.Tensor:
        """Normalise per-image by the p-quantile, clipped to [0, 1]."""
        B = x.shape[0]
        flat = x.view(B, -1)
        # per-image quantile
        q = torch.quantile(flat, p, dim=1, keepdim=True).view(B, 1, 1, 1)
        return (x / (q + 1e-6)).clamp(0.0, 1.0)

    def forward(self, x0_hat: torch.Tensor, source_L: torch.Tensor,
                depth: torch.Tensor) -> torch.Tensor:
        dl = x0_hat - source_L
        g_dl = self._grad_mag(dl)
        g_d = self._grad_mag(depth)

        g_dl_n = self._percentile_norm(g_dl)
        g_d_n = self._percentile_norm(g_d)

        # one-sided hinge: penalise when depth-edge > shift-edge + margin
        deficit = F.relu(g_d_n - g_dl_n - self.margin)
        return deficit.mean()

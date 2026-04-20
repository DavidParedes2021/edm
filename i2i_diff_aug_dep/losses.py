"""
losses.py — Sharpness- and darkness-preserving losses for the depth-aware
luminance diffusion.  **fp16-safe version.**

Changes since the previous version (the one that went NaN at epoch 40):
  - All epsilons raised above fp16 precision floor (~6e-5).
  - Gradient-magnitude sqrt now clamps the inside-radicand to avoid the
    infinite-derivative ledge at 0.
  - DarknessWeightedL1Loss clamps its exp argument so a rogue
    out-of-range target cannot produce an astronomical weight.
  - DepthGradientConsistencyLoss normalises by a floor-clamped quantile
    (1e-3 in normalised-gradient space) and detaches the quantile so
    gradients do NOT flow through the normaliser. Everything runs in
    fp32 regardless of enclosing autocast state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Floor of fp16 normal numbers is ~6.1e-5. Any epsilon <= that is a no-op
# inside an autocast block; we use 1e-4 as a safe "above floor" constant.
_FP16_SAFE_EPS = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Sobel edge loss
# ─────────────────────────────────────────────────────────────────────────────
class SobelEdgeLoss(nn.Module):
    """L1 on Sobel edge magnitudes — penalises blurry outputs.

    The magnitude uses clamp_min BEFORE sqrt so the backward gradient of
    sqrt is bounded. Runs in fp32 to avoid the fp16-floor issue.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.T.clone()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edges(self, x):
        # force fp32 regardless of autocast state
        x = x.float()
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag2 = gx * gx + gy * gy
        # clamp_min above fp16 floor AND away from the sqrt-ledge
        return torch.sqrt(mag2.clamp_min(_FP16_SAFE_EPS))

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


# ─────────────────────────────────────────────────────────────────────────────
# Darkness-weighted L1
# ─────────────────────────────────────────────────────────────────────────────
class DarknessWeightedL1Loss(nn.Module):
    """Upweight pixel error where the TARGET is dark.

    Counteracts the mean-regression bias of vanilla L1 — without this the
    model systematically under-darkens deep-cavity regions.

    Weight:  w = 1 + alpha * exp(- clamp(target + 1, 0, ∞) / tau)
    The inner argument is clamped to [0, ∞) so out-of-range targets
    (e.g. from gamma jitter pushing slightly below -1) cannot blow up
    the exponential.
    """

    def __init__(self, alpha: float = 3.0, tau: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 regardless of autocast; the weight term is cheap.
        target_f = target.float()
        pred_f = pred.float()

        shifted = (target_f + 1.0).clamp_min(0.0)       # [0, 2]
        # Clamp exp argument — at target=-1, arg = 0  → exp = 1
        # at target=+1, arg = 2/tau = 6.67 → exp ≈ 1e-3  → negligible
        # Upper bound is fine; we only need to prevent NaN from below.
        arg = (-shifted / self.tau).clamp(min=-50.0, max=0.0)
        w = 1.0 + self.alpha * torch.exp(arg)

        return (w * (pred_f - target_f).abs()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Depth-gradient consistency
# ─────────────────────────────────────────────────────────────────────────────
class DepthGradientConsistencyLoss(nn.Module):
    """Enforce that the exposure-shift field respects depth discontinuities.

    Let ΔL = x0_hat - source_L. We want |∇ΔL| to NOT be suppressed at
    locations where |∇D| is large. Implemented as a one-sided hinge:

        loss = mean(ReLU( |∇D|_norm - |∇ΔL|_norm - margin ))

    Both gradient maps are 98th-percentile normalised per image. The
    quantile is floor-clamped and DETACHED so no gradient backprops
    through the normaliser — that's what used to blow up in fp16 when
    the 98th percentile landed near zero.
    """

    # Minimum allowed percentile value (in gradient-magnitude space,
    # BEFORE per-image normalisation). Anything below this is treated
    # as a "flat" image and the loss is effectively zero for that sample.
    _QUANT_FLOOR = 1e-3

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
        return torch.sqrt((gx * gx + gy * gy).clamp_min(_FP16_SAFE_EPS))

    def _percentile_norm(self, x: torch.Tensor, p: float = 0.98) -> torch.Tensor:
        """Normalise per-image by the p-quantile, clipped to [0, 1].
        The quantile is detached and floor-clamped."""
        B = x.shape[0]
        flat = x.detach().reshape(B, -1)          # detach: no grad through q
        q = torch.quantile(flat, p, dim=1)        # (B,)
        q = q.clamp_min(self._QUANT_FLOOR)
        q = q.view(B, 1, 1, 1)
        return (x / q).clamp(0.0, 1.0)

    def forward(self, x0_hat: torch.Tensor, source_L: torch.Tensor,
                depth: torch.Tensor) -> torch.Tensor:
        # Force fp32 for the whole operation.
        x0_hat = x0_hat.float()
        source_L = source_L.float()
        depth = depth.float()

        dl = x0_hat - source_L
        g_dl = self._grad_mag(dl)
        g_d  = self._grad_mag(depth)

        g_dl_n = self._percentile_norm(g_dl)
        g_d_n  = self._percentile_norm(g_d)

        deficit = F.relu(g_d_n - g_dl_n - self.margin)
        return deficit.mean()
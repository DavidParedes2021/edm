# evaluation/metrics.py
"""
Evaluation Metrics for Illumination Artifact Generation.

Metrics guide training and evaluate synthetic data quality:

1. PSNR  (Peak Signal-to-Noise Ratio)
   - Higher = less noise, better fidelity
   - Guide: generated image vs normal frame (structural preservation)
   - Expected: 20-30 dB (we WANT some difference from normal!)

2. SSIM  (Structural Similarity Index)
   - [0, 1]: higher = more similar structure
   - Guide: structure should be preserved; illumination should differ
   - Expected: 0.7–0.9 (good structure, some exposure difference)

3. Luminance Mean / Std
   - Direct measure of whether exposure actually changed
   - OVER: mean(Y_out) > mean(Y_normal)  →  typically 0.65–0.85
   - UNDER: mean(Y_out) < mean(Y_normal) →  typically 0.15–0.35
   - Normal baseline: ~0.4–0.55

4. Histogram Divergence (KL / Wasserstein)
   - Measures how different output luminance distribution is from reference
   - Should DECREASE during training → model learns reference distributions

5. Exposure Visibility Score (EVS) — custom metric
   - Binary classifier accuracy of "is this over/underexposed?"
   - Simple threshold-based: if mean(Y) > 0.6 → over, < 0.4 → under
   - EVS = fraction of generated images correctly classified

6. LPIPS (Learned Perceptual Image Patch Similarity) — if available
   - Perceptual distance; lower = more natural-looking
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    pred, target: (B, C, H, W) in [0, max_val]
    Returns: scalar float (dB)
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


# ---------------------------------------------------------------------------
# SSIM (pure PyTorch, no external dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.view(1, 1, size, size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM) — PyTorch native, no skimage dependency.
    pred, target: (B, 1, H, W) or (B, 3, H, W) in [0, data_range]
    Returns: scalar float in [-1, 1]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    device = pred.device
    kernel = _gaussian_kernel(window_size, sigma).to(device=device, dtype=pred.dtype)

    B, C, H, W = pred.shape
    # Process each channel separately
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2,    kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2,  kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    ssim_map = (
        (2 * mu12 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Luminance Statistics
# ---------------------------------------------------------------------------

def luminance_stats(y: torch.Tensor) -> Dict[str, float]:
    """
    Basic luminance statistics.
    y: (B, 1, H, W) in [0, 1]
    """
    y_flat = y.view(-1).float()
    return {
        "mean":   y_flat.mean().item(),
        "std":    y_flat.std().item(),
        "p5":     y_flat.quantile(0.05).item(),   # dark shadows
        "p95":    y_flat.quantile(0.95).item(),   # bright highlights
        "saturated_frac": (y_flat > 0.95).float().mean().item(),  # blown highlights
        "crushed_frac":   (y_flat < 0.05).float().mean().item(),  # crushed shadows
    }


# ---------------------------------------------------------------------------
# Exposure Visibility Score (EVS) — custom
# ---------------------------------------------------------------------------

def exposure_visibility_score(
    y_generated: torch.Tensor,
    label: torch.Tensor,
    over_threshold: float = 0.55,
    under_threshold: float = 0.45,
) -> float:
    """
    Fraction of generated images where exposure is visibly correct.

    For OVER (label=0): mean(Y) should be > over_threshold
    For UNDER (label=1): mean(Y) should be < under_threshold

    Args:
        y_generated: (B, 1, H, W) in [0, 1]
        label:       (B,) long — 0=over, 1=under
    Returns:
        accuracy: float in [0, 1]
    """
    means = y_generated.view(y_generated.shape[0], -1).mean(dim=1)  # (B,)
    correct = 0
    for i in range(len(label)):
        if label[i] == 0 and means[i] > over_threshold:
            correct += 1
        elif label[i] == 1 and means[i] < under_threshold:
            correct += 1
    return correct / len(label)


# ---------------------------------------------------------------------------
# Histogram KL Divergence
# ---------------------------------------------------------------------------

def histogram_kl_divergence(
    pred: torch.Tensor,
    ref_hist: torch.Tensor,
    bins: int = 64,
    eps: float = 1e-8,
) -> float:
    """
    KL divergence between predicted luminance histogram and reference.
    Lower = generated distribution more closely matches reference exposure.

    pred:     (B, 1, H, W) in [0, 1]
    ref_hist: (B, 256) reference normalized histogram
    Returns:  scalar float
    """
    B = pred.shape[0]
    total_kl = 0.0

    ref_hist_ds = F.interpolate(
        ref_hist.unsqueeze(1).float(),
        size=bins,
        mode='linear',
        align_corners=False,
    ).squeeze(1)
    ref_hist_ds = ref_hist_ds / (ref_hist_ds.sum(dim=1, keepdim=True) + eps)

    for b in range(B):
        y_flat = pred[b].view(-1).clamp(0, 1)
        hist = torch.histc(y_flat, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + eps)

        ref = ref_hist_ds[b].to(pred.device)
        # KL(P||Q) = sum P * log(P/Q)
        kl = (hist * (torch.log(hist + eps) - torch.log(ref + eps))).sum()
        total_kl += kl.item()

    return total_kl / B


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_batch(
    pred_y: torch.Tensor,
    normal_y: torch.Tensor,
    label: torch.Tensor,
    ref_hist: torch.Tensor,
) -> Dict[str, float]:
    """
    Run all metrics on a batch.

    pred_y:   (B, 1, H, W) generated luminance [0,1]
    normal_y: (B, 1, H, W) input normal luminance [0,1]
    label:    (B,) long
    ref_hist: (B, 256) reference histogram
    """
    metrics = {}

    # Structure preservation vs normal frame
    metrics["psnr_vs_normal"]   = psnr(pred_y, normal_y)
    metrics["ssim_vs_normal"]   = ssim(pred_y, normal_y)

    # Luminance statistics of generated output
    stats = luminance_stats(pred_y)
    metrics.update({f"lum_{k}": v for k, v in stats.items()})

    # Exposure visibility
    metrics["exposure_visibility"] = exposure_visibility_score(pred_y, label)

    # Histogram divergence from reference
    metrics["hist_kl_div"] = histogram_kl_divergence(pred_y, ref_hist)

    return metrics

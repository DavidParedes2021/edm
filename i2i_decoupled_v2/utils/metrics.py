"""
utils/metrics.py
-----------------
Evaluation metrics for illumination quality assessment.

Metrics
-------
1. PSNR  – structural accuracy (dB); higher is better.
2. SSIM  – perceptual structural similarity [0,1]; higher is better.
3. LPIPS – learned perceptual distance; lower is better.
4. Mean-L shift (ΔL) – average brightness difference in L channel.
5. Histogram distance (Wasserstein-1 / Earth Mover's Distance) between L
   histograms of generated vs. target; lower is better.
6. Exposure correctness: fraction of pixels that fall in expected luminance
   range for the target class.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio.  pred, target: uint8 HxWx3 or HxW."""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(max_val / np.sqrt(mse))


# ---------------------------------------------------------------------------
# SSIM  (pure-numpy, no external dependency)
# ---------------------------------------------------------------------------

def ssim(
    pred:   np.ndarray,
    target: np.ndarray,
    win:    int   = 11,
    sigma:  float = 1.5,
) -> float:
    """SSIM for grayscale or RGB float/uint8 images.  Returns [0,1]."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    pred   = pred.astype(np.float64)
    target = target.astype(np.float64)

    if pred.ndim == 3:
        return np.mean([ssim(pred[..., c], target[..., c]) for c in range(pred.shape[2])])

    from scipy.ndimage import gaussian_filter
    k = win // 2
    mu1 = gaussian_filter(pred,   sigma)
    mu2 = gaussian_filter(target, sigma)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2

    sig1_sq = gaussian_filter(pred   * pred,   sigma) - mu1_sq
    sig2_sq = gaussian_filter(target * target, sigma) - mu2_sq
    sig12   = gaussian_filter(pred   * target, sigma) - mu12

    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return float(np.mean(num / (den + 1e-8)))


# ---------------------------------------------------------------------------
# Histogram distance (Wasserstein-1)
# ---------------------------------------------------------------------------

def histogram_wasserstein(
    pred_L:   np.ndarray,   # H×W float in [0,100]
    target_L: np.ndarray,   # H×W float in [0,100]
    bins:     int = 256,
) -> float:
    """
    Earth Mover's Distance between L-channel histograms.
    Lower → generated exposure distribution matches target better.
    """
    from scipy.stats import wasserstein_distance
    p_hist, bin_edges = np.histogram(pred_L.ravel(),   bins=bins, range=(0, 100), density=True)
    t_hist, _          = np.histogram(target_L.ravel(), bins=bins, range=(0, 100), density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return float(wasserstein_distance(bin_centres, bin_centres, p_hist, t_hist))


# ---------------------------------------------------------------------------
# Brightness shift
# ---------------------------------------------------------------------------

def mean_L_shift(pred_L: np.ndarray, target_L: np.ndarray) -> float:
    """Signed mean-L difference (generated − target).  Near 0 is best."""
    return float(np.mean(pred_L) - np.mean(target_L))


# ---------------------------------------------------------------------------
# Exposure correctness
# ---------------------------------------------------------------------------

def exposure_correctness(L_norm: np.ndarray, class_id: int) -> float:
    """
    Fraction of pixels in the correct luminance half for the target class.
    class_id: 0=over (expect bright, L_norm > 0), 1=under (expect dark, L_norm < 0).
    L_norm in [-1,1].
    """
    if class_id == 0:   # overexposed → pixels should be bright
        return float(np.mean(L_norm > 0.0))
    else:               # underexposed → pixels should be dark
        return float(np.mean(L_norm < 0.0))


# ---------------------------------------------------------------------------
# LPIPS placeholder (uses VGG features, no lpips package required)
# ---------------------------------------------------------------------------

def lpips_vgg(
    pred:   np.ndarray,   # uint8 HxWx3
    target: np.ndarray,   # uint8 HxWx3
    device: torch.device  = torch.device("cpu"),
) -> float:
    """
    Approximate LPIPS using VGG-16 relu3_3 features.
    Lower is better (perceptually more similar).
    """
    from torchvision import models, transforms

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from PIL import Image
    pred_t   = preprocess(Image.fromarray(pred)).unsqueeze(0).to(device)
    target_t = preprocess(Image.fromarray(target)).unsqueeze(0).to(device)

    vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
    with torch.no_grad():
        f_pred   = vgg(pred_t)
        f_target = vgg(target_t)

    dist = F.mse_loss(f_pred, f_target).item()
    return float(dist)


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    pred_rgb:   np.ndarray,   # uint8 HxWx3
    target_rgb: np.ndarray,   # uint8 HxWx3  (real exposure sample)
    normal_rgb: np.ndarray,   # uint8 HxWx3  (input normal)
    class_id:   int,
    device:     torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    Returns a dict of metric_name → value.
    """
    from skimage.color import rgb2lab

    def to_L(rgb):
        lab = rgb2lab(rgb.astype(np.float32) / 255.0)
        return lab[:, :, 0]   # [0, 100]

    L_pred   = to_L(pred_rgb)
    L_target = to_L(target_rgb)
    L_normal = to_L(normal_rgb)

    # Normalised (for exposure_correctness)
    L_pred_norm = (L_pred / 50.0) - 1.0

    metrics = {
        "psnr_vs_target":       psnr(pred_rgb, target_rgb),
        "ssim_vs_normal":       ssim(pred_rgb, normal_rgb),
        "ssim_vs_target":       ssim(pred_rgb, target_rgb),
        "hist_wasserstein":     histogram_wasserstein(L_pred, L_target),
        "mean_L_shift":         mean_L_shift(L_pred, L_target),
        "mean_L_pred":          float(np.mean(L_pred)),
        "mean_L_target":        float(np.mean(L_target)),
        "exposure_correctness": exposure_correctness(L_pred_norm, class_id),
    }

    # LPIPS (optional, slow)
    try:
        metrics["lpips_approx"] = lpips_vgg(pred_rgb, target_rgb, device)
    except Exception:
        metrics["lpips_approx"] = float("nan")

    return metrics

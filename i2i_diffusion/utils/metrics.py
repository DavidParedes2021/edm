"""
utils/metrics.py
----------------
Evaluation metrics for the illumination artifact pipeline.

Metrics computed
-----------------
1. FID           – Fréchet Inception Distance between generated and real domains
2. lpips_source  – Perceptual distance from generated image to source Normal
                   (structural drift indicator; should stay low)
3. ssim_gradient – Structural SSIM on Sobel maps (geometry preservation)
4. highlight_clip_rate – % pixels > 245/255 in generated overexposed images
                         (compared to real Domain B statistics)
5. dark_snr      – Signal-to-noise ratio in dark regions of underexposed
                   images (compared to real Domain C statistics)

FID implementation
-------------------
We use a pure-torch InceptionV3 feature extractor (from torchvision 0.12)
instead of the pytorch-fid library to avoid extra dependencies.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from losses.perceptual  import VGGPerceptualLoss
from losses.ssim_loss   import GradientSSIMLoss


# ── InceptionV3 feature extractor for FID ────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """Extracts 2048-dim pool3 features from InceptionV3."""

    def __init__(self) -> None:
        super().__init__()
        inception = tvm.inception_v3(pretrained=True, transform_input=False)
        # keep up to avgpool, discard classifier
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InceptionV3 expects [0,1] input; resize to 299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x + 1.0) / 2.0   # [-1,1] → [0,1]
        return self.features(x).flatten(1)   # (B, 2048)


def _compute_fid(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    # matrix square root via SVD (avoids scipy dependency)
    covmean_sq = sigma1 @ sigma2
    U, S, Vh = np.linalg.svd(covmean_sq)
    S_sqrt = np.sqrt(np.maximum(S, eps))
    covmean = U @ np.diag(S_sqrt) @ Vh

    tr_covmean = np.trace(covmean)
    return float(
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * tr_covmean
    )


# ── public metric API ─────────────────────────────────────────────────────────

class IlluminationMetrics:
    """
    All evaluation metrics in one object.

    Parameters
    ----------
    device : torch.device
    """

    def __init__(self, device: torch.device) -> None:
        self.dev     = device
        self.inception = InceptionFeatureExtractor().to(device).eval()
        self.vgg_loss  = VGGPerceptualLoss().to(device).eval()
        self.ssim_loss = GradientSSIMLoss().to(device).eval()

    # ── FID ───────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_inception_features(
        self, images: List[torch.Tensor]
    ) -> np.ndarray:
        """
        images : list of (B, 3, H, W) tensors in [-1, 1]
        Returns (N, 2048) numpy array of features.
        """
        feats = []
        for batch in images:
            feats.append(self.inception(batch.to(self.dev)).cpu().numpy())
        return np.concatenate(feats, axis=0)

    def compute_fid(
        self,
        real_features: np.ndarray,    # (N_real, 2048)
        fake_features: np.ndarray,    # (N_fake, 2048)
    ) -> float:
        mu_r, sig_r = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu_f, sig_f = fake_features.mean(0), np.cov(fake_features, rowvar=False)
        return _compute_fid(mu_r, sig_r, mu_f, sig_f)

    # ── perceptual distance to source ─────────────────────────────────────────

    @torch.no_grad()
    def lpips_source(
        self,
        generated: torch.Tensor,   # (B, 3, H, W) [-1,1]
        source:    torch.Tensor,   # (B, 3, H, W) [-1,1]
    ) -> float:
        return self.vgg_loss(
            generated.to(self.dev), source.to(self.dev)
        ).item()

    # ── structural SSIM on gradient maps ─────────────────────────────────────

    @torch.no_grad()
    def ssim_gradient(
        self,
        generated: torch.Tensor,
        source:    torch.Tensor,
    ) -> float:
        """Returns SSIM (higher=more structurally similar). 1 - loss."""
        loss = self.ssim_loss(
            generated.to(self.dev), source.to(self.dev)
        )
        return float(1.0 - loss.item())

    # ── exposure-specific metrics ─────────────────────────────────────────────

    @staticmethod
    def highlight_clip_rate(
        images: torch.Tensor,   # (B, 3, H, W) in [-1, 1]
        threshold: float = 0.92,  # ≈ 235/255 normalised to [-1,1]
    ) -> float:
        """
        Fraction of pixels with intensity > threshold.
        threshold=0.92 corresponds to pixel value 235/255 ≈ 245 after
        denormalising.  Compare this value between generated overexposed
        images and real Domain B to verify physical plausibility.
        """
        # to [0, 1]
        x = (images.float() * 0.5 + 0.5).clamp(0, 1)
        # luminance
        lum = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        t_01 = (threshold + 1.0) / 2.0   # map threshold back to [0,1]
        return float((lum > t_01).float().mean().item())

    @staticmethod
    def dark_region_snr(
        images: torch.Tensor,   # (B, 3, H, W) in [-1, 1]
        dark_threshold: float = -0.6,   # pixels < this are "dark"
    ) -> float:
        """
        Signal-to-noise ratio (dB) in dark regions.
        SNR = 20 * log10(mean_signal / std_noise)
        where signal is the mean pixel value and noise is the std within
        the dark region.  Lower SNR in underexposed images = more noise,
        which matches real Domain C statistics.
        """
        x   = images.float()
        lum = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        # [-1, 1] → lum threshold
        mask    = (lum < dark_threshold)
        if mask.sum() == 0:
            return float("nan")
        dark_px = lum[mask]
        signal  = dark_px.abs().mean().item()
        noise   = dark_px.std().item() + 1e-8
        return float(20 * np.log10(signal / noise))

    # ── summary helper ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_batch(
        self,
        generated_over:  torch.Tensor,   # (B, 3, H, W)
        generated_under: torch.Tensor,
        source_normal:   torch.Tensor,
        real_over:       torch.Tensor,
        real_under:      torch.Tensor,
    ) -> dict:
        """
        Compute all metrics for a single evaluation batch.
        For FID use collect_inception_features + compute_fid over the
        full evaluation set.
        """
        return {
            "lpips_over":      self.lpips_source(generated_over,  source_normal),
            "lpips_under":     self.lpips_source(generated_under, source_normal),
            "ssim_grad_over":  self.ssim_gradient(generated_over,  source_normal),
            "ssim_grad_under": self.ssim_gradient(generated_under, source_normal),
            "clip_rate_gen":   self.highlight_clip_rate(generated_over),
            "clip_rate_real":  self.highlight_clip_rate(real_over),
            "snr_gen":         self.dark_region_snr(generated_under),
            "snr_real":        self.dark_region_snr(real_under),
        }

# utils/losses.py
"""
Multi-term Loss for Illumination Diffusion.

Addresses the two main failure modes:
  1. Blurriness → Perceptual (VGG) loss + Gradient Difference loss
  2. Imperceptible exposure → Histogram Matching loss + Luminance Range loss

Loss terms:
  L_total = λ_l1 * L_l1
           + λ_percep * L_perceptual   (VGG feature distance)
           + λ_grad * L_gradient       (edge/sharpness penalty)
           + λ_hist * L_histogram      (exposure distribution matching)

All losses operate in luminance space [0, 1].
VGG requires 3-channel input → we tile Y → (Y, Y, Y).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Gradient Difference Loss — sharpness preservation
# ---------------------------------------------------------------------------

class GradientDifferenceLoss(nn.Module):
    """
    Penalizes smooth, blurry predictions by comparing image gradients.
    Computes Sobel-like finite differences in x and y directions.
    A sharp prediction has gradients close to the target; blurry ones don't.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) predicted luminance
            target: (B, 1, H, W) target luminance
        """
        def gradient(x):
            # Central differences
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        pred_dx, pred_dy     = gradient(pred)
        target_dx, target_dy = gradient(target)

        loss = (
            F.l1_loss(pred_dx, target_dx) +
            F.l1_loss(pred_dy, target_dy)
        )
        return loss


# ---------------------------------------------------------------------------
# Histogram Matching Loss — exposure control
# ---------------------------------------------------------------------------

class HistogramMatchingLoss(nn.Module):
    """
    Measures distribution distance between predicted and reference luminance.

    Uses soft, differentiable histogram via Gaussian kernel smoothing.
    This directly optimizes for VISIBLE exposure changes, solving the
    "imperceptible exposure" issue by forcing the output luminance
    distribution to match the reference exposure distribution.
    """
    def __init__(self, bins: int = 64, sigma: float = 0.02):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        # Fixed bin centers [0, 1]
        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, bins).view(1, -1),  # (1, bins)
        )

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable histogram via Gaussian kernel.
        x: (B, 1, H, W) in [0, 1]
        Returns: (B, bins) normalized histogram
        """
        B = x.shape[0]
        x_flat = x.view(B, -1, 1)                       # (B, N, 1)
        centers = self.bin_centers.unsqueeze(0)          # (1, 1, bins)
        # Gaussian kernel: exp(-||x - c||^2 / (2σ²))
        weights = torch.exp(
            -((x_flat - centers) ** 2) / (2 * self.sigma ** 2)
        )                                                # (B, N, bins)
        hist = weights.sum(dim=1)                        # (B, bins)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist

    def forward(
        self,
        pred: torch.Tensor,
        ref_hist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:     (B, 1, H, W) predicted luminance in [0, 1]
            ref_hist: (B, 256) pre-computed reference histogram
                      We'll downsample to self.bins via interpolation.
        """
        # Downsample reference histogram to our bin count
        ref_hist_ds = F.interpolate(
            ref_hist.unsqueeze(1).float(),        # (B, 1, 256)
            size=self.bins,
            mode='linear',
            align_corners=False,
        ).squeeze(1)                               # (B, bins)
        ref_hist_ds = ref_hist_ds / (ref_hist_ds.sum(dim=1, keepdim=True) + 1e-8)

        pred_hist = self._soft_histogram(pred.clamp(0, 1))

        # Earth Mover's Distance approximation via CDF distance (Wasserstein-1)
        pred_cdf = torch.cumsum(pred_hist, dim=1)
        ref_cdf  = torch.cumsum(ref_hist_ds, dim=1)
        loss = F.l1_loss(pred_cdf, ref_cdf)
        return loss


# ---------------------------------------------------------------------------
# Perceptual Loss (VGG) — sharpness + semantic fidelity
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps.
    Compatible with torchvision==0.12.0.

    Operates on grayscale luminance by tiling: Y → (Y, Y, Y).
    Feature layers used: relu1_2, relu2_2, relu3_3 (multi-scale).
    Lower layers → low-level sharpness; higher → semantic structure.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import vgg16
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except ImportError:
            # torchvision < 0.13 — fall back to deprecated pretrained flag
            try:
                vgg = vgg16(pretrained=True)  # noqa: TOR001
            except Exception:
                print("[VGG] Warning: pretrained weights unavailable, using random init.")
                vgg = vgg16(pretrained=False)  # noqa: TOR001
        except Exception:
            print("[VGG] Warning: pretrained weights unavailable, using random init.")
            vgg = vgg16(weights=None)

        # Extract feature layers up to relu3_3 (index 16)
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:4])   # relu1_2
        self.slice2 = nn.Sequential(*features[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*features[9:16]) # relu3_3

        # Freeze VGG — we only use it as a fixed feature extractor
        for param in self.parameters():
            param.requires_grad_(False)

        # Move to device immediately
        self.to(device)

        # ImageNet normalization (VGG expects this)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet stats. x: (B, 3, H, W) in [0, 1]."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) in [0, 1]
            target: (B, 1, H, W) in [0, 1]
        """
        # Tile Y channel → 3-channel for VGG
        pred_3   = pred.expand(-1, 3, -1, -1)    # (B, 3, H, W)
        target_3 = target.expand(-1, 3, -1, -1)

        # Ensure minimum VGG input size (32x32)
        if pred_3.shape[-1] < 32:
            pred_3   = F.interpolate(pred_3,   size=(32, 32), mode='bilinear', align_corners=False)
            target_3 = F.interpolate(target_3, size=(32, 32), mode='bilinear', align_corners=False)

        pred_3   = self._normalize(pred_3)
        target_3 = self._normalize(target_3)

        loss = 0.0
        # Slice 1
        f_pred   = self.slice1(pred_3)
        f_target = self.slice1(target_3)
        loss += F.l1_loss(f_pred, f_target)
        # Slice 2
        f_pred   = self.slice2(f_pred)
        f_target = self.slice2(f_target)
        loss += F.l1_loss(f_pred, f_target)
        # Slice 3
        f_pred   = self.slice3(f_pred)
        f_target = self.slice3(f_target)
        loss += F.l1_loss(f_pred, f_target)

        return loss / 3.0


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class IlluminationDiffusionLoss(nn.Module):
    """
    Combined loss for training the illumination diffusion model.

    Note on loss application:
      The DDPM denoising loss (predicting noise ε) is the PRIMARY objective.
      Perceptual and histogram losses are applied in PIXEL SPACE on x_0
      reconstructed from the predicted noise — this guides the model
      toward realistic exposure outputs.
    """
    def __init__(
        self,
        device: torch.device,
        l1_weight: float     = 1.0,
        perceptual_weight: float = 0.5,
        gradient_weight: float   = 0.1,
        histogram_weight: float  = 0.3,
    ):
        super().__init__()
        self.l1_weight          = l1_weight
        self.perceptual_weight  = perceptual_weight
        self.gradient_weight    = gradient_weight
        self.histogram_weight   = histogram_weight

        self.grad_loss = GradientDifferenceLoss()
        self.hist_loss = HistogramMatchingLoss()

        # VGG only if perceptual weight > 0
        self.vgg_loss: Optional[VGGPerceptualLoss] = None
        if perceptual_weight > 0:
            self.vgg_loss = VGGPerceptualLoss(device)

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        ref_hist: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred_noise:   (B, 1, H, W) predicted noise ε_θ
            target_noise: (B, 1, H, W) actual noise ε
            pred_x0:      (B, 1, H, W) reconstructed clean luminance [0,1]
            target_x0:    (B, 1, H, W) clean luminance target [0,1]
            ref_hist:     (B, 256) reference frame histogram

        Returns:
            dict with 'total' loss and individual components
        """
        losses = {}

        # 1. Primary DDPM noise prediction loss (MSE for DDPM is standard;
        #    we use L1 for robustness against outliers)
        losses["l1"] = F.l1_loss(pred_noise, target_noise)

        # 2. Gradient difference (sharpness) — applied on reconstructed x0
        losses["gradient"] = self.grad_loss(pred_x0, target_x0)

        # 3. Histogram matching (exposure control) — against REFERENCE distribution
        losses["histogram"] = self.hist_loss(pred_x0, ref_hist)

        # 4. Perceptual loss (VGG) — structure and texture sharpness
        if self.vgg_loss is not None and self.perceptual_weight > 0:
            losses["perceptual"] = self.vgg_loss(pred_x0, target_x0)
        else:
            losses["perceptual"] = torch.tensor(0.0, device=pred_noise.device)

        # Weighted total
        losses["total"] = (
            self.l1_weight         * losses["l1"]
            + self.gradient_weight   * losses["gradient"]
            + self.histogram_weight  * losses["histogram"]
            + self.perceptual_weight * losses["perceptual"]
        )
        return losses
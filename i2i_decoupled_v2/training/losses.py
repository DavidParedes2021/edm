"""
training/losses.py
-------------------
All loss functions used during training.

Loss taxonomy
-------------
1. Diffusion loss  (L_diff)
   Standard MSE on predicted vs. actual noise.  This is the core signal.
   Improved variant: we also compute MSE in x0-space (predicted clean image)
   for a more direct learning signal.

2. Perceptual loss  (L_perc)
   VGG-16 feature-space distance between predicted x0 and target x0.
   Forces the model to produce perceptually sharp outputs at the feature
   level, not just pixel level.
   WHY THIS FIXES BLURRINESS: MSE in pixel space minimises squared error by
   averaging; perceptual loss penalises blurring at the semantic level where
   averaging is more visually damaging.

3. Exposure loss  (L_exp)
   Histogram-based penalty that compares the mean and std of the predicted
   L channel against the target exposure class statistics (over/under).
   This is the primary mechanism to enforce *strong* exposure changes.
   We add a brightness penalty: predicted mean L must be ≥ (≤) a threshold
   for overexposure (underexposure).

4. Structure loss  (L_struct)
   SSIM computed in the L channel between predicted x0 and the input
   normal-L.  We want illumination to change but structure to be preserved.
   SSIM penalises blur and contrast loss jointly.

5. Adversarial / histogram alignment (optional, off by default):
   Could add a discriminator; we use histogram matching instead since
   training a GAN on top of DDPM is complex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ---------------------------------------------------------------------------
# Perceptual loss (VGG-16)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 features extracted at relu1_2, relu2_2,
    relu3_3 (shallow to mid-level features capture sharpness well).

    Input: single-channel [B,1,H,W] luminance images.
           Replicated to 3 channels internally.

    VGG weights are loaded lazily on first forward pass to allow the object
    to be constructed without a network connection (useful for unit tests and
    offline environments). On GPU machines the weights download automatically
    from PyTorch Hub on the first call to forward().
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self._device    = device
        self._vgg_ready = False
        self.slice1 = self.slice2 = self.slice3 = None

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _load_vgg(self):
        """Lazy-load VGG weights on first use."""
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        except Exception:
            # Fallback for older torchvision (< 0.13)
            vgg = models.vgg16(pretrained=True).features.eval()  # type: ignore[attr-defined]
        for p in vgg.parameters():
            p.requires_grad_(False)
        # Layers: relu1_2=idx4, relu2_2=idx9, relu3_3=idx16
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).to(self._device)
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9]).to(self._device)
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]).to(self._device)
        self._vgg_ready = True

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """L channel [-1,1] → normalised RGB-like [B,3,H,W]."""
        x = (x + 1.0) / 2.0               # → [0,1]
        x = x.repeat(1, 3, 1, 1)           # → [B,3,H,W]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._vgg_ready:
            self._load_vgg()

        pred_v   = self._prep(pred)
        target_v = self._prep(target)

        loss = torch.tensor(0.0, device=pred.device)
        for sl in [self.slice1, self.slice2, self.slice3]:
            pred_v   = sl(pred_v)
            target_v = sl(target_v)
            loss    += F.l1_loss(pred_v, target_v.detach())

        return loss


# ---------------------------------------------------------------------------
# Exposure loss
# ---------------------------------------------------------------------------

class ExposureLoss(nn.Module):
    """
    Enforces strong, class-specific illumination shift.

    For each sample we compute:
      mean_L  = mean of predicted L in [0,100] space (we have [-1,1])
      std_L   = std of predicted L

    Over-exposed target (class=0):
      - mean_L should be > over_mean_threshold
      - Penalise if too dark

    Under-exposed target (class=1):
      - mean_L should be < under_mean_threshold
      - Penalise if too bright

    We also match first two moments to the target's statistics, which
    provides a differentiable histogram-alignment proxy.
    """

    OVER_THRESHOLD  =  0.35   # L normalised [-1,1]; > 0.35 → bright
    UNDER_THRESHOLD = -0.25   # L normalised [-1,1]; < -0.25 → dark

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred:       torch.Tensor,   # [B, 1, H, W] predicted L (normalised)
        target_L:   torch.Tensor,   # [B, 1, H, W] real target L (normalised)
        class_labels: torch.Tensor, # [B] 0=over, 1=under
    ) -> torch.Tensor:
        B = pred.shape[0]
        loss = torch.tensor(0.0, device=pred.device)

        for b in range(B):
            p   = pred[b]           # [1, H, W]
            tgt = target_L[b]       # [1, H, W]
            cl  = class_labels[b].item()

            p_mean = p.mean()
            t_mean = tgt.mean()
            p_std  = p.std()
            t_std  = tgt.std()

            # Moment matching: align predicted mean + std to real target stats
            moment_loss = (p_mean - t_mean) ** 2 + (p_std - t_std) ** 2

            # Directional brightness penalty
            if cl == 0:   # overexposed: must be bright
                dir_penalty = F.relu(self.OVER_THRESHOLD - p_mean)
            else:         # underexposed: must be dark
                dir_penalty = F.relu(p_mean - self.UNDER_THRESHOLD)

            loss += moment_loss + 2.0 * dir_penalty

        return loss / B


# ---------------------------------------------------------------------------
# SSIM-based structure loss
# ---------------------------------------------------------------------------

def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Differentiable SSIM.  x,y ∈ [-1,1].
    Returns SSIM ∈ [-1,1] (we minimise 1 - SSIM).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian kernel
    sigma  = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    gauss  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(gauss, gauss)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, window_size, window_size).expand(x.shape[1], 1, -1, -1)

    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=pad, groups=y.shape[1])

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x * x, kernel, padding=pad, groups=x.shape[1]) - mu_x2
    sig_y2 = F.conv2d(y * y, kernel, padding=pad, groups=y.shape[1]) - mu_y2
    sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=x.shape[1]) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)

    return (numerator / (denominator + 1e-8)).mean()


class StructureLoss(nn.Module):
    """
    SSIM loss between predicted L and the input normal L.
    Encourages structural preservation (edges, textures) despite exposure shift.
    """
    def forward(self, pred_L: torch.Tensor, normal_L: torch.Tensor) -> torch.Tensor:
        ssim_val = _ssim(pred_L, normal_L)
        return 1.0 - ssim_val   # minimise → maximise SSIM


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Combines all losses with configurable weights.
    """

    def __init__(
        self,
        device:            torch.device,
        lambda_diffusion:  float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_exposure:   float = 0.5,
        lambda_structure:  float = 0.2,
    ):
        super().__init__()
        self.w_diff  = lambda_diffusion
        self.w_perc  = lambda_perceptual
        self.w_exp   = lambda_exposure
        self.w_struc = lambda_structure

        self.perc_loss   = VGGPerceptualLoss(device)
        self.exp_loss    = ExposureLoss()
        self.struc_loss  = StructureLoss()

    def forward(
        self,
        noise_pred:    torch.Tensor,    # [B,1,H,W] predicted noise
        noise_target:  torch.Tensor,    # [B,1,H,W] actual noise
        x0_pred:       torch.Tensor,    # [B,1,H,W] reconstructed x0 from noise_pred
        x0_target:     torch.Tensor,    # [B,1,H,W] clean target L (ground truth)
        normal_L:      torch.Tensor,    # [B,1,H,W] input normal L
        class_labels:  torch.Tensor,    # [B]
    ) -> dict:
        # 1. Diffusion MSE (noise space)
        l_diff = F.mse_loss(noise_pred, noise_target)

        # 2. Perceptual (x0 space) – fixes blurriness
        #    Wrapped in try/except so training continues even if VGG weights
        #    are not yet downloaded (first run, offline env, etc.)
        try:
            with torch.cuda.amp.autocast(enabled=False):
                l_perc = self.perc_loss(x0_pred.float(), x0_target.float())
        except Exception:
            l_perc = torch.tensor(0.0, device=noise_pred.device)

        # 3. Exposure (x0 space) – fixes weak exposure changes
        l_exp = self.exp_loss(x0_pred, x0_target, class_labels)

        # 4. Structure (x0 vs normal L) – preserves anatomy
        l_struc = self.struc_loss(x0_pred, normal_L)

        total = (
            self.w_diff  * l_diff  +
            self.w_perc  * l_perc  +
            self.w_exp   * l_exp   +
            self.w_struc * l_struc
        )

        return {
            "total":      total,
            "diffusion":  l_diff.item(),
            "perceptual": l_perc.item() if hasattr(l_perc, 'item') else float(l_perc),
            "exposure":   l_exp.item(),
            "structure":  l_struc.item(),
        }

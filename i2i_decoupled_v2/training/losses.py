"""
training/losses.py  (v2 - fixed)
----------------------------------
Fixes vs v1:
  1. VGG forward: each feature level now receives fresh inputs from the
     original image, not the output of the previous layer. This is the
     correct multi-scale perceptual loss formulation.
  2. Exposure loss: added a much stronger mean-L penalty with a steeper
     margin to force the model to produce clearly over/under-exposed output.
  3. TotalLoss: auxiliary losses (perceptual, exposure, structure) are
     gated by the SNR mask — only applied when t < T*0.35 where
     x0_pred is actually recoverable and gradients are non-zero.
  4. Diffusion MSE is SNR-weighted via per-sample snr_w from GaussianDiffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ---------------------------------------------------------------------------
# VGG Perceptual Loss  (FIXED: independent slices, not cascaded)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss.

    FIXED: Each feature level is extracted from the ORIGINAL input by
    running a fresh forward pass through progressively deeper sub-networks,
    not by re-feeding the output of the previous slice. This is correct
    multi-scale perceptual loss.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self._device    = device
        self._vgg_ready = False
        # Will be set on first call
        self.net_to_relu1_2 = None   # depth-4  features
        self.net_to_relu2_2 = None   # depth-9  features
        self.net_to_relu3_3 = None   # depth-16 features

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _load_vgg(self):
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        except Exception:
            vgg = models.vgg16(pretrained=True).features.eval()  # type: ignore
        for p in vgg.parameters():
            p.requires_grad_(False)
        children = list(vgg.children())
        # Three independent sub-networks, each starting from the input
        self.net_to_relu1_2 = nn.Sequential(*children[:4]).to(self._device)
        self.net_to_relu2_2 = nn.Sequential(*children[:9]).to(self._device)
        self.net_to_relu3_3 = nn.Sequential(*children[:16]).to(self._device)
        self._vgg_ready = True

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0           # [-1,1] → [0,1]
        x = x.repeat(1, 3, 1, 1)      # 1-ch → 3-ch
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._vgg_ready:
            self._load_vgg()

        p = self._prep(pred)
        t = self._prep(target.detach())   # target has no grad

        loss = torch.tensor(0.0, device=pred.device)
        # Each net runs independently from the original input
        for net in [self.net_to_relu1_2, self.net_to_relu2_2, self.net_to_relu3_3]:
            loss = loss + F.l1_loss(net(p), net(t))

        return loss


# ---------------------------------------------------------------------------
# Exposure Loss  (strengthened)
# ---------------------------------------------------------------------------

class ExposureLoss(nn.Module):
    """
    Enforces strong, class-directed exposure shift.

    Two components:
    A) Moment matching: (mean_pred - mean_target)² + (std_pred - std_target)²
       Differentiable histogram alignment proxy.

    B) Directional hard margin:
       over:  max(0, threshold_over  - mean_pred)²   ← penalise if too dark
       under: max(0, mean_pred - threshold_under)²    ← penalise if too bright
       Using SQUARED penalty (was linear before) for stronger gradient signal.
    """

    # In normalised L space ([-1, 1]), overexposed is bright (>0.3), underexposed is dark (<-0.2)
    THRESHOLD_OVER  =  0.25   # pred mean must exceed this for overexposed
    THRESHOLD_UNDER = -0.20   # pred mean must be below this for underexposed
    MARGIN_WEIGHT   =  5.0    # multiplier for directional penalty (was 2.0)

    def forward(
        self,
        pred:          torch.Tensor,    # [B,1,H,W] predicted L (normalised)
        target_L:      torch.Tensor,    # [B,1,H,W] real target L
        class_labels:  torch.Tensor,    # [B]
    ) -> torch.Tensor:
        B    = pred.shape[0]
        loss = torch.tensor(0.0, device=pred.device)

        for b in range(B):
            p   = pred[b]
            tgt = target_L[b]
            cl  = int(class_labels[b].item())

            p_mean = p.mean()
            t_mean = tgt.mean()
            p_std  = p.std()
            t_std  = tgt.std()

            # A) Moment matching
            moment = (p_mean - t_mean) ** 2 + (p_std - t_std) ** 2

            # B) Directional margin (squared for stronger gradients)
            if cl == 0:  # overexposed → must be bright
                margin = F.relu(self.THRESHOLD_OVER - p_mean) ** 2
            else:        # underexposed → must be dark
                margin = F.relu(p_mean - self.THRESHOLD_UNDER) ** 2

            loss = loss + moment + self.MARGIN_WEIGHT * margin

        return loss / B


# ---------------------------------------------------------------------------
# SSIM-based structure loss
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def _ssim_map(x: torch.Tensor, y: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    k  = _gaussian_kernel(win, sigma, x.device).view(1, 1, win, win).expand(x.shape[1], 1, -1, -1)
    p  = win // 2

    mu_x  = F.conv2d(x, k, padding=p, groups=x.shape[1])
    mu_y  = F.conv2d(y, k, padding=p, groups=y.shape[1])
    mu_x2 = mu_x ** 2;  mu_y2 = mu_y ** 2;  mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x*x, k, padding=p, groups=x.shape[1]) - mu_x2
    sig_y2 = F.conv2d(y*y, k, padding=p, groups=y.shape[1]) - mu_y2
    sig_xy = F.conv2d(x*y, k, padding=p, groups=x.shape[1]) - mu_xy

    num = (2*mu_xy + C1) * (2*sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return num / (den + 1e-8)


class StructureLoss(nn.Module):
    """1 − SSIM between predicted x0 and the normal-L input. Preserves anatomy."""
    def forward(self, pred_L: torch.Tensor, normal_L: torch.Tensor) -> torch.Tensor:
        return 1.0 - _ssim_map(pred_L, normal_L).mean()


# ---------------------------------------------------------------------------
# Composite loss  (FIXED: SNR-gated aux losses)
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Combined loss with SNR-aware gating.

    Auxiliary losses (perceptual, exposure, structure) are only computed
    for samples where the timestep t is in the low-noise regime
    (t < T * aux_loss_t_frac, typically t < 350 for T=1000).

    At high timesteps, x0_pred = (x_t - sqrt(1-alpha)*eps_pred) / sqrt(alpha)
    has a near-zero denominator — clamping to [-1,1] produces zero gradients,
    so these losses are useless and waste compute. Gating removes the waste.

    The diffusion MSE is weighted per-sample by the Min-SNR-γ weight,
    making the model prioritise low-t steps (where image content is visible).
    """

    def __init__(
        self,
        device:            torch.device,
        lambda_diffusion:  float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_exposure:   float = 0.8,   # increased from 0.5
        lambda_structure:  float = 0.2,
    ):
        super().__init__()
        self.w_diff  = lambda_diffusion
        self.w_perc  = lambda_perceptual
        self.w_exp   = lambda_exposure
        self.w_struc = lambda_structure

        self.perc_loss  = VGGPerceptualLoss(device)
        self.exp_loss   = ExposureLoss()
        self.struc_loss = StructureLoss()

    def forward(
        self,
        noise_pred:   torch.Tensor,   # [B,1,H,W]
        noise_target: torch.Tensor,   # [B,1,H,W]
        x0_pred:      torch.Tensor,   # [B,1,H,W]
        x0_target:    torch.Tensor,   # [B,1,H,W]
        normal_L:     torch.Tensor,   # [B,1,H,W]
        class_labels: torch.Tensor,   # [B]
        snr_w:        torch.Tensor,   # [B]   Min-SNR-γ weights
        aux_mask:     torch.Tensor,   # [B] bool — True where t is low (aux losses valid)
    ) -> dict:
        B = noise_pred.shape[0]

        # 1. SNR-weighted diffusion MSE
        per_sample_mse = F.mse_loss(noise_pred, noise_target, reduction='none').mean(dim=(1,2,3))
        l_diff = (snr_w.squeeze() * per_sample_mse).mean()

        # 2–4. Auxiliary losses: only on low-t samples
        n_aux = int(aux_mask.sum().item())

        if n_aux > 0:
            x0p_low  = x0_pred[aux_mask]
            x0t_low  = x0_target[aux_mask]
            norm_low = normal_L[aux_mask]
            cls_low  = class_labels[aux_mask]

            # Perceptual
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    l_perc = self.perc_loss(x0p_low.float(), x0t_low.float())
            except Exception:
                l_perc = torch.tensor(0.0, device=noise_pred.device)

            # Exposure
            l_exp = self.exp_loss(x0p_low, x0t_low, cls_low)

            # Structure
            l_struc = self.struc_loss(x0p_low, norm_low)
        else:
            zero = torch.tensor(0.0, device=noise_pred.device)
            l_perc = l_exp = l_struc = zero

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
            "exposure":   l_exp.item()  if hasattr(l_exp,  'item') else float(l_exp),
            "structure":  l_struc.item() if hasattr(l_struc,'item') else float(l_struc),
            "n_aux_samples": n_aux,
        }
"""
models/losses.py
Hybrid loss = MSE(noise/v) + λ · Perceptual(x0_pred, x0_gt)

Perceptual loss uses VGG16 relu2_2 and relu3_3 features.
VGG is frozen and always on the same device as the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Typing import needed at module level
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# VGG Perceptual loss (torchvision==0.12.0)
# ──────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 intermediate features.
    Layers:  relu1_2  (idx 3),  relu2_2  (idx 8),  relu3_3  (idx 15)

    torchvision 0.12.0 compatible – uses vgg16(pretrained=True).
    """

    def __init__(self):
        super().__init__()
        # Lazy import so CPU-only smoke tests don't fail if torchvision missing
        try:
            import torchvision.models as tvm
            vgg = tvm.vgg16(pretrained=True)
        except Exception as e:
            raise ImportError(
                "torchvision is required for VGG perceptual loss. "
                "Install: pip install torchvision==0.12.0+cu113"
            ) from e

        # Slice feature blocks
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:4])    # relu1_2
        self.slice2 = nn.Sequential(*features[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*features[9:16])  # relu3_3

        for param in self.parameters():
            param.requires_grad = False  # Frozen

        # ImageNet normalisation (VGG expects [0,1] not [-1,1])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] → ImageNet-normalised."""
        x = (x + 1.0) * 0.5        # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, 3, H, W] in [-1, 1]
        Returns scalar perceptual loss.
        """
        pred_n   = self._normalize(pred)
        target_n = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)

        f_pred   = pred_n
        f_target = target_n
        for block in [self.slice1, self.slice2, self.slice3]:
            f_pred   = block(f_pred)
            f_target = block(f_target)
            loss     = loss + F.l1_loss(f_pred, f_target.detach())

        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid training loss
# ──────────────────────────────────────────────────────────────────────────────

class HybridDiffusionLoss(nn.Module):
    """
    L = mse_weight · MSE(model_out, target)
      + perceptual_weight · VGG(x0_pred, x0_gt)

    If perceptual_weight == 0, no VGG is instantiated → safe for CPU debug runs.
    """

    def __init__(
        self,
        mse_weight: float        = 1.0,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight        = mse_weight
        self.perceptual_weight = perceptual_weight

        self.vgg: Optional[VGGPerceptualLoss] = None
        if perceptual_weight > 0.0:
            self.vgg = VGGPerceptualLoss()

    def forward(
        self,
        model_out: torch.Tensor,    # predicted v or ε, [B,3,H,W]
        target: torch.Tensor,       # ground-truth v or ε, [B,3,H,W]
        x0_pred: torch.Tensor,      # decoded x_0 prediction, [B,3,H,W]
        x0_gt: torch.Tensor,        # ground-truth x_0, [B,3,H,W]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, {component losses dict}).
        """
        mse_loss = F.mse_loss(model_out, target)
        total    = self.mse_weight * mse_loss
        log      = {"mse": mse_loss.item()}

        if self.perceptual_weight > 0.0 and self.vgg is not None:
            # Move VGG to same device as inputs (once)
            device = model_out.device
            if next(self.vgg.parameters()).device != device:
                self.vgg = self.vgg.to(device)

            perc_loss = self.vgg(x0_pred, x0_gt)
            total     = total + self.perceptual_weight * perc_loss
            log["perceptual"] = perc_loss.item()

        log["total"] = total.item()
        return total, log

if __name__ == "__main__":
    loss_fn = HybridDiffusionLoss(mse_weight=1.0, perceptual_weight=0.0)
    a = torch.randn(2, 3, 64, 64)
    b = torch.randn(2, 3, 64, 64)
    l, info = loss_fn(a, b, a, b)
    print("Loss:", l.item(), info)

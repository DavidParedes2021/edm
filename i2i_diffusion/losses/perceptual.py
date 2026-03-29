"""
losses/perceptual.py
--------------------
LPIPS-style perceptual loss using VGG-16 feature maps.

Implemented from scratch using torchvision 0.12's VGG16 — no external
lpips library required.  Feature layers match the standard LPIPS Lin
calibration (relu1_2, relu2_2, relu3_3, relu4_3, relu5_3).

Inputs are expected in [-1, 1]; internally rescaled to [0, 1] and then
ImageNet-normalised before passing to VGG.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class VGGPerceptualLoss(nn.Module):
    """
    Multi-layer VGG-16 perceptual loss.

    Parameters
    ----------
    layer_weights : list[float]
        Per-layer weights applied before summing.  Default follows the
        standard LPIPS calibration (heavier on deeper layers).
    """

    # VGG-16 feature layer indices for relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
    _SLICE_ENDS = [4, 9, 16, 23, 30]

    def __init__(
        self,
        layer_weights: List[float] = [0.03125, 0.0625, 0.125, 0.25, 1.0],
    ) -> None:
        super().__init__()

        # Register ImageNet stats as buffers so they follow .to(device/dtype)
        # automatically — avoids the CPU/CUDA device mismatch under AMP.
        self.register_buffer(
            "_vgg_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_vgg_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        vgg = tvm.vgg16(pretrained=True).features
        # split into slices at each target relu
        self.slices = nn.ModuleList()
        prev = 0
        for end in self._SLICE_ENDS:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        # freeze VGG — we never want to update these weights
        for p in self.parameters():
            p.requires_grad_(False)

        self.layer_weights = layer_weights
        assert len(self.layer_weights) == len(self.slices)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] → ImageNet-normalised.  Buffers already on correct device/dtype."""
        x = (x + 1.0) / 2.0                          # → [0, 1]
        mean = self._vgg_mean.to(dtype=x.dtype)
        std  = self._vgg_std.to(dtype=x.dtype)
        return (x - mean) / std

    def forward(
        self,
        pred: torch.Tensor,    # (B, 3, H, W)  generated image
        target: torch.Tensor,  # (B, 3, H, W)  reference (Normal frame)
    ) -> torch.Tensor:
        """Returns scalar perceptual loss."""
        pred_vgg   = self._preprocess(pred)
        target_vgg = self._preprocess(target)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for slice_net, w in zip(self.slices, self.layer_weights):
            pred_vgg   = slice_net(pred_vgg)
            target_vgg = slice_net(target_vgg.detach())
            # normalise features before L2 (matches LPIPS Lin normalisation)
            p = F.normalize(pred_vgg,   dim=1)
            t = F.normalize(target_vgg, dim=1)
            loss = loss + w * F.mse_loss(p, t)

        return loss
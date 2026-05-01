"""
Multi-term loss for the Y-channel diffusion model.

  loss = lambda_eps  * MSE(eps_pred, noise)              # primary DDPM objective
       + lambda_l1   * L1(x0_pred, y_target)             # reconstruction
       + lambda_grad * L1(grad(x0_pred), grad(y_target)) # sharpness
       + lambda_vgg  * VGG(x0_pred, y_target)            # perceptual
       + lambda_hist * SortedW1(x0_pred, y_target)       # 1D Wasserstein on luminance histogram

All auxiliary losses are computed on the *predicted x0*, not the noise, so they
compose with the eps prediction objective. Auxiliaries are cast to fp32 at
entry to keep AMP happy with the older torchvision build (the VGG16 ImageNet
weights mean/std normalization can blow up in fp16 on extreme values).
"""
from __future__ import annotations
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ImageNet stats for VGG normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptual(nn.Module):
    """
    VGG16 perceptual loss; works on Y replicated to 3 channels.

    Compatible with torchvision 0.12 (uses the legacy `pretrained=True` API)
    *and* 0.13+ (falls back to `weights=VGG16_Weights.DEFAULT` if available).
    """
    def __init__(self, layers=(3, 8, 15, 22)):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights  # 0.13+
            vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features
        except (ImportError, AttributeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vgg = torchvision.models.vgg16(pretrained=True).features  # 0.12
        self.layers = list(layers)
        max_layer = max(self.layers) + 1
        self.vgg = nn.Sequential(*list(vgg)[:max_layer])
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.vgg.eval()
        self.register_buffer('_mean', _IMAGENET_MEAN, persistent=False)
        self.register_buffer('_std',  _IMAGENET_STD,  persistent=False)

    def _features(self, x_rgb01: torch.Tensor):
        x = (x_rgb01 - self._mean) / self._std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """y_pred, y_target: (B, 1, H, W) in [-1, 1]."""
        # AMP guard: VGG is in eval/no-grad with fp32 weights; we cast inputs.
        y_pred = y_pred.float()
        y_target = y_target.float()
        # [-1, 1] -> [0, 1] -> RGB triplicate
        a = ((y_pred + 1.0) * 0.5).clamp(0, 1).expand(-1, 3, -1, -1)
        b = ((y_target + 1.0) * 0.5).clamp(0, 1).expand(-1, 3, -1, -1)
        with torch.no_grad():
            feats_b = self._features(b)
        feats_a = self._features(a)
        loss = sum(F.l1_loss(fa, fb) for fa, fb in zip(feats_a, feats_b))
        return loss / len(self.layers)


def gradient_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between image gradients. Penalizes blur."""
    pred = pred.float(); target = target.float()
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def sorted_wasserstein1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Closed-form 1D Wasserstein-1 between two sets of pixel intensities,
    via sorting. Computed per-image and averaged.
    """
    pred = pred.float(); target = target.float()
    B = pred.shape[0]
    p = pred.reshape(B, -1).sort(dim=1).values
    q = target.reshape(B, -1).sort(dim=1).values
    return (p - q).abs().mean()


class YCLDILoss(nn.Module):
    def __init__(
        self,
        lambda_eps:  float = 1.0,
        lambda_l1:   float = 0.1,
        lambda_grad: float = 0.1,
        lambda_vgg:  float = 0.05,
        lambda_hist: float = 0.05,
        use_vgg:     bool  = True,
    ):
        super().__init__()
        self.lambda_eps  = lambda_eps
        self.lambda_l1   = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_vgg  = lambda_vgg if use_vgg else 0.0
        self.lambda_hist = lambda_hist
        self.vgg: Optional[VGGPerceptual] = VGGPerceptual() if use_vgg else None

    def forward(
        self,
        eps_pred: torch.Tensor,
        noise:    torch.Tensor,
        x0_pred:  torch.Tensor,    # in [-1, 1]
        y_target: torch.Tensor,    # in [-1, 1]
    ) -> tuple[torch.Tensor, dict]:
        log = {}
        loss = self.lambda_eps * F.mse_loss(eps_pred.float(), noise.float())
        log['eps_mse'] = loss.detach()

        if self.lambda_l1 > 0:
            l1 = F.l1_loss(x0_pred.float(), y_target.float())
            loss = loss + self.lambda_l1 * l1
            log['l1'] = l1.detach()

        if self.lambda_grad > 0:
            gd = gradient_difference_loss(x0_pred, y_target)
            loss = loss + self.lambda_grad * gd
            log['grad'] = gd.detach()

        if self.vgg is not None and self.lambda_vgg > 0:
            v = self.vgg(x0_pred, y_target)
            loss = loss + self.lambda_vgg * v
            log['vgg'] = v.detach()

        if self.lambda_hist > 0:
            h = sorted_wasserstein1(x0_pred, y_target)
            loss = loss + self.lambda_hist * h
            log['hist'] = h.detach()

        log['total'] = loss.detach()
        return loss, log

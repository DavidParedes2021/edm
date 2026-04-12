"""
losses.py — Sharpness-preserving losses for luminance diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SobelEdgeLoss(nn.Module):
    """Penalises blurry outputs by comparing Sobel edge maps."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.T.clone()
        # (out_c, in_c, kH, kW)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edges(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class VGGPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using early VGG16 features.

    Works on single-channel input by repeating to 3 channels.
    Uses only first 2 feature blocks to keep it cheap.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(pretrained=False)
        # try to load pretrained weights; if unavailable, loss still works
        try:
            vgg = models.vgg16(pretrained=True)
        except Exception:
            print("[VGGLoss] pretrained weights unavailable — using random init "
                  "(perceptual loss will still regularise but less effectively)")
        # take features up to relu2_2 (layer index 8)
        self.features = nn.Sequential(*list(vgg.features.children())[:9]).to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )

    def forward(self, pred, target):
        # pred, target: (B, 1, H, W) in [-1, 1]
        # map to [0,1] and replicate to 3ch
        pred3 = ((pred + 1) / 2).repeat(1, 3, 1, 1)
        tgt3 = ((target + 1) / 2).repeat(1, 3, 1, 1)
        pred3 = (pred3 - self.mean) / self.std
        tgt3 = (tgt3 - self.mean) / self.std
        return F.l1_loss(self.features(pred3), self.features(tgt3))

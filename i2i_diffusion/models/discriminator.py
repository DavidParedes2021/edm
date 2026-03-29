"""
models/discriminator.py
-----------------------
PatchGAN discriminator operating on 70×70 receptive-field patches.
One instance is created per target domain (over + under).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN.  Classifies overlapping 70-px patches as real / fake.
    Input: 3-channel image in [-1, 1].
    Output: (B, 1, H', W') map of logits (no sigmoid — use BCEWithLogitsLoss).
    """

    def __init__(self, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(3, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4,
                          stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

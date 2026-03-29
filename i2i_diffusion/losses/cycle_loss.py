"""
losses/cycle_loss.py
--------------------
Cycle-consistency and identity regularisation losses.

For unpaired data, these are the primary anti-hallucination mechanisms:

  Cycle: G_under(G_over(A)) ≈ A   and   G_over(G_under(A)) ≈ A
         Enforces that the translation is *invertible* — the model cannot
         create or destroy scene geometry.

  Identity: G_over(B) ≈ B  (when given an already-overexposed image,
             produce minimal change)
             Prevents the generator from drifting when the input already
             satisfies the target condition.

Both losses use L1 (not MSE) — more tolerant of small photometric
differences in the round-trip that are physically plausible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CycleConsistencyLoss(nn.Module):
    """
    L1 cycle loss:  ||F(G(x)) - x||_1

    where G translates A→B and F translates B→A.
    In our setup, we use the *same* generator with a different class label
    for the reverse pass (over→normal uses label=normal, etc.).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        reconstructed: torch.Tensor,  # G_inv(G(x))
        original:      torch.Tensor,  # x
    ) -> torch.Tensor:
        return F.l1_loss(reconstructed, original.detach())


class IdentityLoss(nn.Module):
    """
    L1 identity loss:  ||G_B(y) - y||_1

    When the generator receives an image already in the target domain,
    it should return it (approximately) unchanged.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        generated: torch.Tensor,  # G_B(y_B)
        target:    torch.Tensor,  # y_B  (real sample from domain B)
    ) -> torch.Tensor:
        return F.l1_loss(generated, target.detach())


class AdversarialLoss(nn.Module):
    """
    Least-squares GAN loss (LSGAN).
    More stable than vanilla BCE for diffusion-based generators.

    generator_loss(fake_logits)  :  generator wants disc to output 1
    discriminator_loss(real, fake):  disc wants real→1, fake→0
    """

    def __init__(self) -> None:
        super().__init__()

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return torch.mean((fake_logits - 1.0) ** 2) * 0.5

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        real_loss = torch.mean((real_logits - 1.0) ** 2) * 0.5
        fake_loss = torch.mean(fake_logits ** 2) * 0.5
        return real_loss + fake_loss

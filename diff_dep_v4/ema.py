"""Exponential moving average of model parameters.

EMA is critical for diffusion sampling quality: the EMA copy is what
we use at inference. The live model continues to receive gradients.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            # Cast in case of dtype mismatch (e.g., fp16 master in some setups)
            ema_p.data.mul_(d).add_(p.data.to(ema_p.data.dtype), alpha=1.0 - d)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.ema_model.load_state_dict(sd)

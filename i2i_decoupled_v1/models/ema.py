# models/ema.py
"""
Exponential Moving Average (EMA) of model weights.
EMA weights produce significantly sharper and more stable inference results
than raw training weights, especially for diffusion models.

Usage:
    ema = EMAModel(model, decay=0.9999)
    # after each optimizer step:
    ema.step(model)
    # for inference:
    with ema.average_parameters():
        output = model(...)
"""
import copy
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Iterable


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    EMA decay of 0.9999 is standard for diffusion models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Deep copy — EMA params live separately from training params
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        # Freeze EMA params (never gradient-updated)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module):
        """Update EMA after each training step."""
        for ema_p, model_p in zip(
            self.shadow.parameters(), model.parameters()
        ):
            # EMA update: shadow = decay * shadow + (1 - decay) * current
            ema_p.data.mul_(self.decay).add_(
                model_p.data, alpha=1.0 - self.decay
            )

    def to(self, device: torch.device):
        """Move EMA model to device."""
        self.shadow = self.shadow.to(device)
        return self

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Context manager: temporarily swap model weights with EMA weights.
        Restores original weights on exit.

        Usage:
            with ema.average_parameters(model):
                out = model(x)
        """
        # Store original params
        original = [p.data.clone() for p in model.parameters()]

        # Load EMA params into model
        for model_p, ema_p in zip(model.parameters(), self.shadow.parameters()):
            model_p.data.copy_(ema_p.data)

        try:
            yield
        finally:
            # Restore original params
            for model_p, orig in zip(model.parameters(), original):
                model_p.data.copy_(orig)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

"""
models/ema.py
-------------
Exponential moving average of model weights.  Used for stable inference
without affecting training dynamics.
"""
from __future__ import annotations

import copy
from typing import Union

import torch
import torch.nn as nn


class EMA:
    """
    Maintains a shadow copy of a model's parameters updated by EMA.

    The shadow is created on CPU and moved to the correct device lazily
    on the first update() call, so it is safe to construct EMA before
    Accelerate / DDP moves the live model to GPU.

    Usage
    -----
        ema = EMA(model, decay=0.9999)
        # after Accelerate prepare() has moved model to GPU:
        ema.to(device)          # optional explicit move
        # after each optimiser step:
        ema.update(model)       # auto-detects device on first call
        # for inference:
        with ema.average_parameters(model):
            output = model(x)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        # Freeze shadow — it is only ever updated via EMA, never by gradients.
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._device: torch.device | None = None

    def to(self, device: torch.device | str) -> "EMA":
        """Explicitly move the shadow model to `device`."""
        self.shadow.to(device)
        self._device = torch.device(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        EMA step.  On the first call the shadow is lazily moved to the
        same device as the live model so construction order doesn't matter.
        """
        # Lazy device sync: happens once, costs one .to() call.
        live_device = next(model.parameters()).device
        if self._device != live_device:
            self.shadow.to(live_device)
            self._device = live_device

        for s_param, m_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
        # Re-sync device after loading in case the checkpoint was saved from
        # a different device than the current run.
        if self._device is not None:
            self.shadow.to(self._device)

    # ── context manager: swap live weights with EMA weights for inference ──

    class _SwapContext:
        def __init__(self, model: nn.Module, shadow: nn.Module) -> None:
            self._model  = model
            self._shadow = shadow
            self._backup: dict = {}

        def __enter__(self) -> nn.Module:
            # Clone live weights as backup (keep them on their current device)
            self._backup = {
                k: v.clone() for k, v in self._model.state_dict().items()
            }
            # Load shadow weights onto the live model; map to live model's
            # device so there is no cross-device assignment.
            live_device = next(self._model.parameters()).device
            shadow_state = {
                k: v.to(live_device)
                for k, v in self._shadow.state_dict().items()
            }
            self._model.load_state_dict(shadow_state)
            return self._model

        def __exit__(self, *args) -> None:
            self._model.load_state_dict(self._backup)

    def average_parameters(self, model: nn.Module) -> "_SwapContext":
        return self._SwapContext(model, self.shadow)
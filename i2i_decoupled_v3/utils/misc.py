"""
utils/misc.py
--------------
Utility functions: EMA, optimiser, scheduler, checkpointing, seed.
"""

import os
import copy
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains EMA of model weights.
    Used for inference: EMA model is more stable than the live training model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module):
        """Swap EMA weights into model for inference."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.decay  = state["decay"]


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float = 1e-4) -> AdamW:
    # Separate weight decay: don't decay biases / norm params
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params,    "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_steps:  int,
    total_steps:   int,
) -> SequentialLR:
    """Linear warmup → cosine annealing."""
    warmup = LinearLR(
        optimizer,
        start_factor = 1e-6,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max  = max(1, total_steps - warmup_steps),
        eta_min= 1e-6,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:        Path,
    model:       nn.Module,
    ema:         EMA,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    epoch:       int,
    global_step: int,
    loss:        float,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":       epoch,
            "global_step": global_step,
            "loss":        loss,
            "model":       model.state_dict(),
            "ema":         ema.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
        },
        str(path),
    )


def load_checkpoint(
    path:          Path,
    model:         nn.Module,
    ema:           Optional[EMA]                     = None,
    optimizer:     Optional[torch.optim.Optimizer]   = None,
    scheduler      = None,
    device:        torch.device                      = torch.device("cpu"),
) -> dict:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')})")
    return ckpt

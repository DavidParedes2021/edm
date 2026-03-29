"""
models/controlnet_lite.py
-------------------------
A lightweight structural-hint injector that conditions the U-Net decoder
on an edge / gradient map extracted from the Normal source frame.

Design rationale
----------------
Full ControlNet (Zhang et al. 2023) duplicates the entire U-Net encoder.
That requires ~8 GB extra VRAM for a standard 512-resolution model —
too expensive for a single shared GPU.

We use a *thin* alternative:
  - A small CNN encoder processes the 1-channel edge map and produces
    additive residuals for the first N encoder blocks of the U-Net.
  - Residuals are injected via PyTorch forward hooks registered on the
    U-Net's down_blocks.
  - The injector is detached from the U-Net's parameters so it can be
    frozen or trained independently.

This gives the geometry-preservation benefit of ControlNet at ~1/8 the
parameter cost.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_groups(num_channels: int, preferred: int = 8) -> int:
    """
    Return the largest divisor of `num_channels` that is <= `preferred`.
    Prevents the 'num_channels must be divisible by num_groups' ValueError
    that occurs when num_channels (e.g. 1 or 4) is smaller than preferred.
    Falls back to 1 (equivalent to LayerNorm per-channel) if nothing fits.
    """
    for g in range(preferred, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # always safe


# ── small residual CNN ─────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _safe_groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ControlNetLite(nn.Module):
    """
    Lightweight structural hint injector.

    Takes a 1-channel edge map [B, 1, H, W] in [-1, 1] and produces
    `num_layers` residual tensors whose channel counts match the first
    `num_layers` encoder blocks of the U-Net.

    Parameters
    ----------
    in_channels : int
        Input channels of the edge map (1).
    block_out_channels : tuple[int]
        Channel widths of the U-Net encoder blocks we want to influence.
        Only the first `num_layers` entries are used.
    num_layers : int
        How many encoder blocks to inject into (≤ len(block_out_channels)).
    """

    def __init__(
        self,
        in_channels:       int = 1,
        block_out_channels: tuple = (128, 256, 512, 512),
        num_layers:        int = 3,
    ) -> None:
        super().__init__()

        self.num_layers = min(num_layers, len(block_out_channels))
        channels        = block_out_channels[:self.num_layers]

        # stem: map 1-ch edge to first block width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(channels[0]), channels[0]),
            nn.SiLU(),
        )

        # per-block heads: residual encoder + 1×1 projection
        self.heads = nn.ModuleList()
        for i, ch in enumerate(channels):
            layers: list[nn.Module] = [_ResBlock(channels[0])]
            if i > 0:
                # downsample + project to match deeper block channels
                layers += [
                    nn.Conv2d(channels[0], ch, 3, stride=2**i, padding=1, bias=False),
                    nn.GroupNorm(_safe_groups(ch), ch),
                    nn.SiLU(),
                ]
            else:
                layers += [nn.Conv2d(channels[0], ch, 1, bias=False)]
            self.heads.append(nn.Sequential(*layers))

        # learnable zero initialisation so early training is stable
        for head in self.heads:
            last_conv = [m for m in head.modules() if isinstance(m, nn.Conv2d)][-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, edge_map: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns a list of `num_layers` residual tensors.
        Each tensor has shape (B, C_i, H_i, W_i) matching the
        corresponding U-Net encoder block's output shape.
        """
        feat = self.stem(edge_map)
        residuals = []
        for head in self.heads:
            residuals.append(head(feat))
        return residuals


# ── hook-based injection context ──────────────────────────────────────────────

class ControlNetHookContext:
    """
    Injects ControlNetLite residuals into a ClassConditionedUNet's
    down_blocks via registered forward hooks.  Use as a context manager
    so hooks are always cleaned up.

    Usage
    -----
        residuals = controlnet(edge_map)
        with ControlNetHookContext(unet.unet, residuals):
            noise_pred = unet(noisy, source, timestep, labels)
    """

    def __init__(
        self,
        unet_model: nn.Module,       # the inner diffusers UNet2DConditionModel
        residuals: List[torch.Tensor],
    ) -> None:
        self._unet      = unet_model
        self._residuals = residuals
        self._hooks: list = []

    def __enter__(self) -> "ControlNetHookContext":
        down_blocks = list(self._unet.down_blocks)
        for i, (block, res) in enumerate(
            zip(down_blocks, self._residuals)
        ):
            # capture res in closure
            def make_hook(r: torch.Tensor):
                def hook(module, inputs, output):
                    # output may be a tuple (hidden, attn_weights) for attn blocks
                    if isinstance(output, tuple):
                        hidden = output[0]
                        # only add if spatial dims match (they should)
                        if hidden.shape == r.shape:
                            return (hidden + r,) + output[1:]
                        return output
                    if output.shape == r.shape:
                        return output + r
                    return output
                return hook

            h = block.register_forward_hook(make_hook(res))
            self._hooks.append(h)
        return self

    def __exit__(self, *args) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

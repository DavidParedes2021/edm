"""Generic utilities: seeding, config loading, sample grids, path derivation."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def derive_output_paths(cfg: dict) -> dict[str, Path]:
    """Return {root, ckpt, samples, logs, generated} as Path objects (created)."""
    root = Path(cfg["output"]["root"])
    paths = {
        "root": root,
        "ckpt": root / "checkpoints",
        "samples": root / "samples",
        "logs": root / "logs",
        "generated": root / "generated",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_sample_grid(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    """Save a grid of (N, C, H, W) images in [0, 1] to ``path``.

    Robust to single-sample batches (does not error on N == 1).
    """
    if images.dim() == 3:                 # (C, H, W) -> (1, C, H, W)
        images = images.unsqueeze(0)
    if images.dim() != 4:
        raise ValueError(f"images must be 3-D or 4-D; got {tuple(images.shape)}")
    if images.shape[0] == 0:
        return
    nrow_eff = max(1, min(nrow, int(images.shape[0])))
    grid = make_grid(images.detach().clamp(0.0, 1.0), nrow=nrow_eff, padding=2)
    arr = (grid.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def humanize_param_count(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)

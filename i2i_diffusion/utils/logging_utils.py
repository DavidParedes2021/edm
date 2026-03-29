"""
utils/logging_utils.py
----------------------
wandb integration, local image-grid logging, and checkpoint management.
Compatible with wandb 0.14.2.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

log = logging.getLogger(__name__)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1,1] → uint8 numpy (H, W, C)."""
    t = t.detach().cpu().float()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class Logger:
    """
    Wraps wandb and local file logging.

    Parameters
    ----------
    cfg         : full config dict
    accelerator : Accelerate accelerator (used to check is_main_process)
    """

    def __init__(self, cfg: dict, accelerator) -> None:
        self.cfg  = cfg
        self.acc  = accelerator
        self._run = None

        lc = cfg["logging"]
        tc = cfg["training"]

        if self.acc.is_main_process and lc["wandb"]["enabled"]:
            try:
                import wandb
                init_kwargs: dict = {
                    "project":   lc["wandb"]["project"],
                    "name":      tc["run_name"],
                    "config":    cfg,
                    "resume":    "allow",
                }
                if lc["wandb"].get("entity"):
                    init_kwargs["entity"] = lc["wandb"]["entity"]
                self._run = wandb.init(**init_kwargs)
                log.info("wandb initialised.")
            except Exception as e:
                log.warning(f"wandb init failed ({e}) — logging locally only.")

        # local image log directory
        self._img_dir = (
            Path(tc["output_dir"]) / tc["run_name"] / "image_logs"
        )
        self._img_dir.mkdir(parents=True, exist_ok=True)

    # ── scalar logging ────────────────────────────────────────────────────────

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        if not self.acc.is_main_process:
            return
        msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        log.info(f"[step {step}]  {msg}")
        if self._run is not None:
            self._run.log(metrics, step=step)

    # ── image logging ─────────────────────────────────────────────────────────

    def log_image_grid(
        self,
        normal:  torch.Tensor,   # (N, 3, H, W) [-1,1]
        over:    torch.Tensor,
        under:   torch.Tensor,
        epoch:   int,
        n_imgs:  int = 4,
    ) -> None:
        """Saves a side-by-side grid: Normal | Over | Under."""
        if not self.acc.is_main_process:
            return

        n = min(n_imgs, normal.shape[0])
        rows = []
        for i in range(n):
            rows.append(
                torch.stack([normal[i], over[i], under[i]])  # (3, 3, H, W)
            )
        grid = vutils.make_grid(
            torch.cat(rows, dim=0),
            nrow=3,
            normalize=True,
            value_range=(-1, 1),
        )

        # save locally
        img_path = self._img_dir / f"epoch_{epoch:04d}.png"
        vutils.save_image(grid, img_path)

        # log to wandb
        if self._run is not None:
            import wandb
            self._run.log(
                {"samples": wandb.Image(str(img_path))},
                step=epoch,
            )

    # ── close ─────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

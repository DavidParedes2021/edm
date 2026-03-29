"""
data/dataset.py
---------------
UnpairedIlluminationDataset  –  yields (normal, over, under) from three
independent, unsorted directories.  At training time the model sees each
domain independently; at inference time only normal frames are consumed.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_transforms, histogram_match_pil


# ── helpers ───────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return paths


# ── dataset ───────────────────────────────────────────────────────────────────

class UnpairedIlluminationDataset(Dataset):
    """
    Returns a dict with keys:
        normal  – Tensor[3, H, W] in [-1, 1]
        over    – Tensor[3, H, W] in [-1, 1]
        under   – Tensor[3, H, W] in [-1, 1]
        normal_edge – Tensor[1, H, W] in [-1, 1]  (structural hint for ControlNet)
        label_over  – int tensor 0
        label_under – int tensor 1
    The three domains are sampled *independently* (unpaired).
    """

    LABEL_OVER  = 0
    LABEL_UNDER = 1

    def __init__(
        self,
        normal_dir: str,
        over_dir: str,
        under_dir: str,
        image_size: int = 256,
        augment: bool = True,
        hist_match_warmstart: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.normal_paths = _collect_images(normal_dir)
        self.over_paths   = _collect_images(over_dir)
        self.under_paths  = _collect_images(under_dir)

        self.image_size = image_size
        self.hist_match = hist_match_warmstart

        # longest domain sets epoch length
        self.length = max(
            len(self.normal_paths),
            len(self.over_paths),
            len(self.under_paths),
        )

        self.tf_image = build_transforms(image_size, augment=augment)
        self.tf_edge  = build_transforms(image_size, augment=augment, edge=True)

        self._rng = random.Random(seed)

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _idx(self, paths: List[Path], i: int) -> Path:
        """Cycle through a list so all domains have the same apparent length."""
        return paths[i % len(paths)]

    # ── public ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        normal_img = self._load(self._idx(self.normal_paths, idx))

        # shuffle over/under indices so over[i] ≠ normal[i] (unpaired)
        over_idx  = self._rng.randint(0, len(self.over_paths)  - 1)
        under_idx = self._rng.randint(0, len(self.under_paths) - 1)

        over_img  = self._load(self.over_paths[over_idx])
        under_img = self._load(self.under_paths[under_idx])

        # optional histogram-matching warm-start
        if self.hist_match:
            over_img  = histogram_match_pil(normal_img, over_img)
            under_img = histogram_match_pil(normal_img, under_img)

        normal_t     = self.tf_image(normal_img)
        over_t       = self.tf_image(over_img)
        under_t      = self.tf_image(under_img)
        normal_edge  = self.tf_edge(normal_img)   # 1-channel edge map

        return {
            "normal":       normal_t,
            "over":         over_t,
            "under":        under_t,
            "normal_edge":  normal_edge,
            "label_over":   torch.tensor(self.LABEL_OVER,  dtype=torch.long),
            "label_under":  torch.tensor(self.LABEL_UNDER, dtype=torch.long),
        }


class NormalOnlyDataset(Dataset):
    """Lightweight dataset used at inference time (only normal frames needed)."""

    def __init__(self, normal_dir: str, image_size: int = 256) -> None:
        super().__init__()
        self.paths = _collect_images(normal_dir)
        self.tf_image = build_transforms(image_size, augment=False)
        self.tf_edge  = build_transforms(image_size, augment=False, edge=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img  = Image.open(self.paths[idx]).convert("RGB")
        return {
            "normal":      self.tf_image(img),
            "normal_edge": self.tf_edge(img),
            "path":        str(self.paths[idx]),
        }

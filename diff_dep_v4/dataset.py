"""Endoscopy luminance dataset for class-conditional diffusion training.

The dataset returns:
    - y     : (1, H, W) float32 in [-1, 1] -- the Y channel of the frame
    - label : long scalar tensor in {0, 1, 2}

Class layout (must stay in sync with model.py / train.py / infer.py):
    0 = normal
    1 = overexposed
    2 = underexposed
    3 = NULL  (CFG only -- never produced by the dataset)
"""
from __future__ import annotations

from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from colorspace import split_ycbcr


CLASS_NORMAL = 0
CLASS_OVER = 1
CLASS_UNDER = 2
CLASS_NULL = 3   # reserved for CFG; never returned by the dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(root: str | Path) -> list[Path]:
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Image directory not found: {p}")
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not files:
        raise RuntimeError(f"No images found under {p}")
    return sorted(files)


class YCbCrEndoscopyDataset(Dataset):
    """Loads (Y, class_label) pairs from three directories of unpaired frames."""

    def __init__(
        self,
        normal_dir: str,
        over_dir: str,
        under_dir: str,
        image_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.image_size = int(image_size)
        self.augment = bool(augment)

        self.entries: list[tuple[Path, int]] = []
        for p in _list_images(normal_dir):
            self.entries.append((p, CLASS_NORMAL))
        for p in _list_images(over_dir):
            self.entries.append((p, CLASS_OVER))
        for p in _list_images(under_dir):
            self.entries.append((p, CLASS_UNDER))

        if not self.entries:
            raise RuntimeError("Dataset is empty after listing all class directories.")

        counts = {CLASS_NORMAL: 0, CLASS_OVER: 0, CLASS_UNDER: 0}
        for _, c in self.entries:
            counts[c] += 1
        self._counts = counts

    # -------- interface --------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def class_counts(self) -> dict[int, int]:
        return dict(self._counts)

    # -------- loading ----------------------------------------------------

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        # Square center crop on the shorter side, then resize. This is the
        # safest default for endoscopy frames (keeps the circular FOV centred).
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return TF.to_tensor(img)  # (3, H, W) in [0, 1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, label = self.entries[idx]
        rgb = self._load_rgb(path)  # (3, H, W)

        if self.augment and random.random() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])

        # split_ycbcr expects (B, 3, H, W); add a fake batch dim.
        Y_norm, _ = split_ycbcr(rgb.unsqueeze(0))
        Y_norm = Y_norm.squeeze(0)        # (1, H, W) in [-1, 1]

        return {
            "y": Y_norm.contiguous(),
            "label": torch.tensor(label, dtype=torch.long),
        }

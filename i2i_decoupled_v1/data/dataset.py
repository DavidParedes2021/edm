# data/dataset.py
"""
Unpaired Illumination Dataset for Diffusion Training.

Dataset structure:
  data/normal/  → 504 normal frames  (our X: condition input + ground truth)
  data/over/    → 281 overexposed frames  (unpaired reference statistics)
  data/under/   → 817 underexposed frames (unpaired reference statistics)

Training strategy:
  Since data is UNPAIRED, we train a class-conditional model:
    - Input:  Normal frame Y channel (condition)
    - Target: Reference exposure statistics (histogram matching target)
    - Label:  Exposure class (OVER or UNDER)

  The model learns:  p(Y_exposed | Y_normal, class)
  At inference: given a normal frame → generate over AND under versions

Reference frames are used ONLY for:
  1. Computing histogram statistics (training loss target)
  2. Statistical data augmentation of the luminance distribution

The diffusion model's denoising target is learned from the DISTRIBUTION
of reference frames, not pixel-level correspondence.
"""
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from utils.color_space import rgb_to_ycbcr, normalize_luminance


class IlluminationDataset(Dataset):
    """
    Training dataset.

    Each sample contains:
      - normal_y_norm: (1, H, W) luminance of normal frame, normalized [-1, 1]
      - normal_y:      (1, H, W) luminance of normal frame, [0, 1]
      - normal_rgb:    (3, H, W) RGB of normal frame, [0, 1]
      - ref_y:         (1, H, W) luminance of a reference (over or under) frame
      - label:         int — 0=over, 1=under
      - ref_hist:      (256,) float — luminance histogram of reference frame
    """

    LABEL_OVER  = 0
    LABEL_UNDER = 1

    def __init__(
        self,
        normal_path: str,
        over_path:   str,
        under_path:  str,
        image_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.split = split

        normal_dir = Path(normal_path)
        over_dir   = Path(over_path)
        under_dir  = Path(under_path)

        # Validate directories
        for d, name in [(normal_dir, "normal_path"), (over_dir, "over_path"), (under_dir, "under_path")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found: {d}\n"
                    f"Check the '{name}' entry in your config file."
                )

        # Load file lists
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        normal_files = sorted([f for f in normal_dir.iterdir() if f.suffix.lower() in exts])
        over_files   = sorted([f for f in over_dir.iterdir()   if f.suffix.lower() in exts])
        under_files  = sorted([f for f in under_dir.iterdir()  if f.suffix.lower() in exts])

        if len(normal_files) == 0:
            raise RuntimeError(f"No images found in {normal_dir}")

        print(f"[Dataset] Found: {len(normal_files)} normal, "
              f"{len(over_files)} over, {len(under_files)} under")

        # Train/val split on normal frames (deterministic)
        rng = random.Random(seed)
        normal_shuffled = normal_files.copy()
        rng.shuffle(normal_shuffled)
        n_val = max(1, int(len(normal_shuffled) * val_fraction))
        if split == "val":
            self.normal_files = normal_shuffled[:n_val]
        else:
            self.normal_files = normal_shuffled[n_val:]

        self.over_files  = over_files
        self.under_files = under_files

        # Transforms
        self.transform = self._build_transform(image_size, augment=(split == "train"))

        print(f"[Dataset] {split}: {len(self.normal_files)} normal frames | "
              f"{len(self.over_files)} over ref | {len(self.under_files)} under ref")

    def _build_transform(self, size: int, augment: bool):
        ops = [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # → [0, 1]
        ]
        if augment:
            ops.insert(1, transforms.RandomHorizontalFlip())
        return transforms.Compose(ops)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        """Load image as (3, H, W) float tensor in [0, 1]."""
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # (3, H, W)

    @staticmethod
    def _compute_hist(y: torch.Tensor, bins: int = 256) -> torch.Tensor:
        """
        Compute normalized luminance histogram.
        y: (1, H, W) in [0, 1]
        Returns: (bins,) float tensor
        """
        y_flat = y.view(-1).clamp(0, 1)
        hist = torch.histc(y_flat, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)  # normalize
        return hist

    def __len__(self) -> int:
        return len(self.normal_files)

    def __getitem__(self, idx: int) -> dict:
        # ---- Normal frame ----
        normal_rgb = self._load_rgb(self.normal_files[idx])    # (3, H, W)
        normal_ycbcr = rgb_to_ycbcr(normal_rgb.unsqueeze(0)).squeeze(0)  # (3, H, W)
        normal_y = normal_ycbcr[:1, :, :]                      # (1, H, W) in [0,1]
        normal_y_norm = normalize_luminance(normal_y)           # (1, H, W) in [-1,1]

        # ---- Reference frame (randomly sample from over OR under) ----
        # Each sample randomly picks one exposure type
        if random.random() < 0.5:
            label = self.LABEL_OVER
            ref_path = random.choice(self.over_files)
        else:
            label = self.LABEL_UNDER
            ref_path = random.choice(self.under_files)

        ref_rgb  = self._load_rgb(ref_path)
        ref_ycbcr = rgb_to_ycbcr(ref_rgb.unsqueeze(0)).squeeze(0)
        ref_y    = ref_ycbcr[:1, :, :]                          # (1, H, W)
        ref_hist = self._compute_hist(ref_y)                    # (256,)

        return {
            "normal_y_norm": normal_y_norm,     # (1, H, W) diffusion input
            "normal_y":      normal_y,          # (1, H, W) [0,1] for loss
            "normal_rgb":    normal_rgb,        # (3, H, W) for visualization
            "ref_y":         ref_y,             # (1, H, W) reference luminance
            "ref_hist":      ref_hist,          # (256,) histogram target
            "label":         torch.tensor(label, dtype=torch.long),
        }


def build_dataloader(
    normal_path: str,
    over_path:   str,
    under_path:  str,
    image_size: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> DataLoader:
    dataset = IlluminationDataset(
        normal_path=normal_path,
        over_path=over_path,
        under_path=under_path,
        image_size=image_size,
        split=split,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
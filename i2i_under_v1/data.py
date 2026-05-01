"""
Unpaired YCbCr dataset.

The dataset is the *union* of the normal and underexposed folders. Each sample
carries a class label (0 = normal, 1 = underexposed). The dataloader returns
the Y channel separately so the diffusion UNet can operate on it directly,
while CbCr is kept around for inference recombination.

Note: classification of the source folders is imperfect (per the brief --
some "underexposed" frames are actually fine, etc). The class label is
treated as a soft signal during training; CFG dropout (10%) further
regularizes it.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from color import rgb_to_ycbcr


_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

CLASS_NORMAL = 0
CLASS_UNDER = 1
NUM_REAL_CLASSES = 2  # null class is NUM_REAL_CLASSES (used for CFG dropout)


def _list_images(root: str | None) -> List[str]:
    if not root:
        return []
    p = Path(root)
    if not p.is_dir():
        return []
    out = [str(f) for f in p.rglob('*') if f.suffix.lower() in _IMG_EXTS]
    return sorted(out)


class UnpairedYCbCrDataset(Dataset):
    """Returns dicts with keys: y, cbcr, rgb, cls, path."""

    def __init__(
        self,
        normal_dir: str,
        underexposed_dir: str,
        image_size: int = 256,
        train: bool = True,
        balance: bool = True,
    ):
        self.image_size = image_size
        self.train = train

        normal = _list_images(normal_dir)
        under = _list_images(underexposed_dir)
        if not normal and not under:
            raise FileNotFoundError(
                f"No images found in '{normal_dir}' or '{underexposed_dir}'."
            )

        # The dataset is unbalanced. Up-sample the smaller class for training.
        if train and balance and normal and under:
            n_max = max(len(normal), len(under))
            normal = (normal * ((n_max + len(normal) - 1) // len(normal)))[:n_max]
            under = (under * ((n_max + len(under) - 1) // len(under)))[:n_max]

        self.samples: List[Tuple[str, int]] = (
            [(p, CLASS_NORMAL) for p in normal]
            + [(p, CLASS_UNDER) for p in under]
        )
        random.Random(0).shuffle(self.samples)  # interleave classes

        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # -> (3, H, W) in [0, 1]
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path, cls = self.samples[idx]
        rgb = self.tf(Image.open(path).convert('RGB'))   # (3, H, W) in [0, 1]
        ycbcr = rgb_to_ycbcr(rgb)
        y = ycbcr[0:1].clamp(0.0, 1.0)
        cbcr = ycbcr[1:].clamp(0.0, 1.0)
        return {
            'y': y,
            'cbcr': cbcr,
            'rgb': rgb,
            'cls': torch.tensor(cls, dtype=torch.long),
            'path': path,
        }


def make_loader(cfg: dict, train: bool = True) -> DataLoader:
    ds = UnpairedYCbCrDataset(
        normal_dir=cfg['data']['normal_dir'],
        underexposed_dir=cfg['data']['underexposed_dir'],
        image_size=cfg['data']['image_size'],
        train=train,
        balance=cfg['data'].get('balance', True),
    )
    return DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'] if train else 4,
        shuffle=train,
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=train,
        persistent_workers=cfg['data'].get('num_workers', 4) > 0,
    )

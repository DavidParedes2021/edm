"""
dataset/illumination_dataset.py
--------------------------------
Unpaired illumination dataset.

Design decisions
----------------
* Works in CIE-LAB colour space:  images are loaded as RGB, converted to LAB.
  Only the L channel (luminance) is fed to the diffusion model.  The AB
  channels are stored separately and recombined at inference time.
* Returns (L_normal, L_target, ab_normal, class_label) so the model is
  supervised on luminance only while structure (AB) is preserved.
* Because the dataset is UNPAIRED, normal / over / under images are sampled
  independently – the model must learn the *distribution* of each exposure
  class conditioned on the normal-L input.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helper: RGB → LAB and back
# ---------------------------------------------------------------------------

def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 HxWx3 RGB → float32 HxWx3 LAB (L in [0,100], AB in [-128,127])."""
    from skimage.color import rgb2lab
    return rgb2lab(img_rgb.astype(np.float32) / 255.0).astype(np.float32)


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert float32 HxWx3 LAB → uint8 HxWx3 RGB, clipped to [0,255]."""
    from skimage.color import lab2rgb
    rgb = lab2rgb(img_lab.astype(np.float32))          # → [0, 1]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def normalise_L(L: np.ndarray) -> np.ndarray:
    """Map L channel [0,100] → [-1, 1] for the diffusion model."""
    return (L / 50.0) - 1.0


def denormalise_L(L_norm: np.ndarray) -> np.ndarray:
    """Map model output [-1,1] → [0,100]."""
    return (L_norm + 1.0) * 50.0


def normalise_AB(AB: np.ndarray) -> np.ndarray:
    """Map AB channels [-128,127] → [-1,1]."""
    return AB / 128.0


def denormalise_AB(AB_norm: np.ndarray) -> np.ndarray:
    return AB_norm * 128.0


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def build_spatial_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC)]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IlluminationDataset(Dataset):
    """
    Unpaired dataset for normal → {overexposed, underexposed} translation.

    Class labels
    ------------
    0  →  overexposed target
    1  →  underexposed target

    __getitem__ returns
    -------------------
    dict with keys:
      L_normal   : Tensor [1, H, W], normalised luminance of the normal frame
      AB_normal  : Tensor [2, H, W], normalised AB channels of the normal frame
      L_target   : Tensor [1, H, W], normalised luminance of the paired target
      AB_target  : Tensor [2, H, W], AB channels of the target frame
      class_label: int   (0=over, 1=under)
      normal_path: str
      target_path: str
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
    ):
        self.image_size = image_size
        self.augment    = augment
        self.transform  = build_spatial_transform(image_size, augment)

        self.normal_paths = self._collect(normal_dir)
        self.over_paths   = self._collect(over_dir)
        self.under_paths  = self._collect(under_dir)

        if len(self.normal_paths) == 0:
            raise RuntimeError(f"No images found in normal_dir: {normal_dir}")
        if len(self.over_paths) == 0:
            raise RuntimeError(f"No images found in over_dir: {over_dir}")
        if len(self.under_paths) == 0:
            raise RuntimeError(f"No images found in under_dir: {under_dir}")

        # Each item either uses an over or under target → 2× dataset length
        self.length = len(self.normal_paths) * 2

    @staticmethod
    def _collect(directory: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        paths = [
            str(p)
            for p in sorted(Path(directory).rglob("*"))
            if p.suffix.lower() in exts
        ]
        return paths

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        # First half → over target; second half → under target
        half = len(self.normal_paths)
        if idx < half:
            class_label  = self.LABEL_OVER
            normal_path  = self.normal_paths[idx % len(self.normal_paths)]
            target_path  = random.choice(self.over_paths)
        else:
            class_label  = self.LABEL_UNDER
            normal_path  = self.normal_paths[idx % len(self.normal_paths)]
            target_path  = random.choice(self.under_paths)

        L_n, AB_n = self._load_lab(normal_path)
        L_t, AB_t = self._load_lab(target_path)

        return {
            "L_normal":    torch.from_numpy(L_n).unsqueeze(0),   # [1,H,W]
            "AB_normal":   torch.from_numpy(AB_n).permute(2,0,1),# [2,H,W]
            "L_target":    torch.from_numpy(L_t).unsqueeze(0),   # [1,H,W]
            "AB_target":   torch.from_numpy(AB_t).permute(2,0,1),# [2,H,W]
            "class_label": class_label,
            "normal_path": normal_path,
            "target_path": target_path,
        }

    # ------------------------------------------------------------------
    def _load_lab(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image, apply spatial transforms, convert to LAB, normalise."""
        img = Image.open(path).convert("RGB")
        img = self.transform(img)                       # PIL → PIL (resized)
        img_np = np.array(img)                          # HxWx3 uint8
        lab = rgb_to_lab(img_np)                        # HxWx3 float32

        L  = normalise_L(lab[:, :, 0])                 # HxW
        AB = normalise_AB(lab[:, :, 1:])               # HxWx2

        return L, AB


# ---------------------------------------------------------------------------
# Collate helper for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = vals          # list of strings
    return out

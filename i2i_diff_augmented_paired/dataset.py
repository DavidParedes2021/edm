"""
dataset.py — LAB-space luminance dataset for exposure diffusion.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ---- colour-space helpers ------------------------------------------------- #

def rgb_to_lab_numpy(img_np: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC RGB → float32 HWC LAB.

    L in [0, 100], A in [-128, 127], B in [-128, 127].
    Uses the D65 illuminant (sRGB standard).
    """
    img = img_np.astype(np.float32) / 255.0

    # linearise sRGB
    mask = img > 0.04045
    img = np.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

    # sRGB → XYZ (D65)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = img @ mat.T

    # normalise by D65 white point
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    mask = xyz > eps
    xyz_f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * xyz_f[:, :, 1] - 16.0
    A = 500.0 * (xyz_f[:, :, 0] - xyz_f[:, :, 1])
    B = 200.0 * (xyz_f[:, :, 1] - xyz_f[:, :, 2])

    return np.stack([L, A, B], axis=-1)


def lab_to_rgb_numpy(lab: np.ndarray) -> np.ndarray:
    """Convert float32 HWC LAB → uint8 HWC RGB."""
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)

    xyz = np.stack([xr * 0.95047, yr * 1.00000, zr * 1.08883], axis=-1).astype(np.float32)

    # XYZ → linear sRGB
    mat_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    rgb_lin = xyz @ mat_inv.T

    rgb_lin = np.clip(rgb_lin, 0, None)
    mask = rgb_lin > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_lin ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_lin)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


# ---- dataset -------------------------------------------------------------- #

class ExposureDataset(Dataset):
    """Loads images from multiple domain folders, returns L channel + domain label.

    Domain labels:
        0 = overexposed
        1 = underexposed

    Normal images are **not** used during training (they are inference inputs).
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        overexposed_dir: str,
        underexposed_dir: str,
        image_size: int = 256,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.augment = augment

        over_paths = self._scan(overexposed_dir)
        under_paths = self._scan(underexposed_dir)

        # build (path, label) list
        self.samples = [(p, 0) for p in over_paths] + [(p, 1) for p in under_paths]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {overexposed_dir} or {underexposed_dir}"
            )

        # oversample minority domain so batches are balanced
        n_over, n_under = len(over_paths), len(under_paths)
        if n_over > 0 and n_under > 0:
            majority = max(n_over, n_under)
            if n_over < majority:
                extra = random.choices(
                    [(p, 0) for p in over_paths], k=majority - n_over
                )
                self.samples.extend(extra)
            elif n_under < majority:
                extra = random.choices(
                    [(p, 1) for p in under_paths], k=majority - n_under
                )
                self.samples.extend(extra)

        random.shuffle(self.samples)
        print(f"[Dataset] {len(self.samples)} samples "
              f"(over={n_over}, under={n_under}, balanced to {len(self.samples)})")

    def _scan(self, folder: str):
        folder = Path(folder)
        if not folder.is_dir():
            print(f"[Warning] directory not found: {folder}")
            return []
        return sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # resize (keep square — endoscopy frames are already square)
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        # augmentation
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)

        img_np = np.array(img)
        lab = rgb_to_lab_numpy(img_np)

        # extract L channel, normalise to [-1, 1]
        L = lab[:, :, 0]  # [0, 100]
        L_norm = (L / 50.0) - 1.0  # → [-1, 1]

        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()  # (1, H, W)
        return L_tensor, label


class NormalImageDataset(Dataset):
    """Loads normal images for inference — returns L tensor + full LAB + path."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, normal_dir: str, image_size: int = 256):
        self.image_size = image_size
        folder = Path(normal_dir)
        self.paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in self.EXTENSIONS
        )
        print(f"[NormalDataset] {len(self.paths)} images from {normal_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img)
        lab = rgb_to_lab_numpy(img_np)  # (H, W, 3)

        L = lab[:, :, 0]
        L_norm = (L / 50.0) - 1.0
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()

        # keep AB for recombination
        AB = lab[:, :, 1:]  # (H, W, 2)

        return L_tensor, torch.from_numpy(AB).float(), str(path)

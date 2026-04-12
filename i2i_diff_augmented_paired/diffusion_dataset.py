"""
diffusion_dataset.py — Paired L-channel dataset for conditional diffusion.

Loads (source_L, target_L, domain_label) triplets from the pairs generated
by generate_pairs.py.  Also provides a NormalInferenceDataset for generation.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from exposure_augment import rgb_to_lab


# --------------------------------------------------------------------------- #
# Training dataset: paired (normal L → target L)
# --------------------------------------------------------------------------- #

class PairedLuminanceDataset(Dataset):
    """
    Each item returns:
        source_L : (1, H, W) float32 in [-1, 1]  — normal luminance
        target_L : (1, H, W) float32 in [-1, 1]  — augmented luminance
        label    : int  (0 = overexposed, 1 = underexposed)
    """

    def __init__(
        self,
        pairs_dir: str,
        image_size: int = 256,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.augment = augment
        pairs = Path(pairs_dir)

        # scan normal LAB arrays (contain full LAB)
        normal_dir = pairs / "normal"
        over_dir = pairs / "overexposed"
        under_dir = pairs / "underexposed"

        self.samples = []  # (normal_path, target_path, label)

        # match normal arrays to their overexposed/underexposed targets
        for npy in sorted(normal_dir.glob("*.npy")):
            stem = npy.stem  # e.g. "EDD2020_NAF0088"

            # overexposed targets (may include variants _v0, _v1, ...)
            for target in sorted(over_dir.glob(f"{stem}*.npy")):
                self.samples.append((str(npy), str(target), 0))

            # underexposed targets
            for target in sorted(under_dir.glob(f"{stem}*.npy")):
                self.samples.append((str(npy), str(target), 1))

        if not self.samples:
            raise RuntimeError(
                f"No paired samples found in {pairs_dir}. "
                f"Run generate_pairs.py first."
            )

        # balance domains
        over_samples = [s for s in self.samples if s[2] == 0]
        under_samples = [s for s in self.samples if s[2] == 1]
        n_over, n_under = len(over_samples), len(under_samples)
        majority = max(n_over, n_under)

        if n_over < majority and n_over > 0:
            extra = random.choices(over_samples, k=majority - n_over)
            self.samples.extend(extra)
        elif n_under < majority and n_under > 0:
            extra = random.choices(under_samples, k=majority - n_under)
            self.samples.extend(extra)

        random.shuffle(self.samples)
        print(f"[PairedDataset] {len(self.samples)} pairs "
              f"(over={n_over}, under={n_under}, balanced to {len(self.samples)})")

    @staticmethod
    def _normalise_L(L: np.ndarray) -> np.ndarray:
        """L ∈ [0, 100] → [-1, 1]."""
        return (L / 50.0) - 1.0

    @staticmethod
    def _resize_2d(arr: np.ndarray, size: int) -> np.ndarray:
        """Resize a 2D float array using PIL (Lanczos)."""
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        normal_path, target_path, label = self.samples[idx]

        # load source LAB (H, W, 3)
        normal_lab = np.load(normal_path).astype(np.float32)
        source_L = normal_lab[..., 0]  # (H, W)

        # load target L (H, W)
        target_L = np.load(target_path).astype(np.float32)

        # resize to training resolution
        source_L = self._resize_2d(source_L, self.image_size)
        target_L = self._resize_2d(target_L, self.image_size)

        # augmentation (applied identically to both)
        if self.augment:
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=1).copy()
                target_L = np.flip(target_L, axis=1).copy()
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=0).copy()
                target_L = np.flip(target_L, axis=0).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                source_L = np.rot90(source_L, k).copy()
                target_L = np.rot90(target_L, k).copy()

        # normalise and tensorify
        source_L = self._normalise_L(source_L)
        target_L = self._normalise_L(target_L)

        source_tensor = torch.from_numpy(source_L).unsqueeze(0)  # (1, H, W)
        target_tensor = torch.from_numpy(target_L).unsqueeze(0)  # (1, H, W)

        return source_tensor, target_tensor, label


# --------------------------------------------------------------------------- #
# Inference dataset: loads original RGB normal images
# --------------------------------------------------------------------------- #

class NormalInferenceDataset(Dataset):
    """
    Returns:
        source_L : (1, H, W)   normalised L channel at training resolution
        AB       : (H_orig, W_orig, 2)  original-resolution chrominance
        L_orig   : (H_orig, W_orig)     original-resolution L (for texture recombination)
        path     : str
        orig_hw  : (int, int)  original height, width
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, normal_dir: str, image_size: int = 256):
        self.image_size = image_size
        folder = Path(normal_dir)
        self.paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in self.EXTENSIONS
        )
        print(f"[InferenceDataset] {len(self.paths)} images from {normal_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = np.array(Image.open(path).convert("RGB"))
        lab = rgb_to_lab(img)

        L_orig = lab[..., 0].copy()
        AB = lab[..., 1:].copy()

        # resize L for model input
        L_resized = np.array(
            Image.fromarray(L_orig.astype(np.float32), mode="F").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        L_norm = (L_resized / 50.0) - 1.0
        source_tensor = torch.from_numpy(L_norm).unsqueeze(0)

        return source_tensor, AB, L_orig, str(path), img.shape[:2]

"""
dataset.py — Dataset utilities for the Illumination Diffusion pipeline.

Supports:
  • Unpaired Normal / Overexposed / Underexposed directories.
  • On-the-fly histogram-specification pseudo-pairing (optional, for early
    training signal when paired data is unavailable).
  • Returns (normal_img, target_img, ev_scalar) triplets.

All images are returned as torch.Tensor in [-1, 1] (VAE convention).
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def list_images(directory: str) -> List[Path]:
    """Return sorted list of image paths under *directory*."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in exts)


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """Resize → RGB → Tensor in [-1, 1]."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    t = T.ToTensor()(img)          # [0, 1]
    return t * 2.0 - 1.0           # [-1, 1]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [-1,1] tensor (C,H,W) → PIL Image."""
    t = (t.clamp(-1, 1) + 1.0) / 2.0   # [0, 1]
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Histogram-specification pseudo-pairing
# ──────────────────────────────────────────────────────────────────────────────

def _cdf(arr: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(arr.flatten(), bins=bins, range=(0, 256))
    cdf = hist.cumsum().astype(np.float32)
    cdf /= cdf[-1] + 1e-7
    centers = (edges[:-1] + edges[1:]) / 2.0
    return cdf, centers


def histogram_specification(
    source: np.ndarray,    # uint8 HxWxC  (Normal frame)
    reference: np.ndarray  # uint8 HxWxC  (Overexposed/Underexposed sample)
) -> np.ndarray:
    """
    Warp *source*'s pixel histogram to match *reference*'s histogram,
    channel by channel.  Returns uint8 array same shape as source.

    Used to create pseudo-paired training targets from unpaired data.
    """
    result = np.empty_like(source)
    for c in range(source.shape[2]):
        src_cdf, _   = _cdf(source[:, :, c])
        ref_cdf, ref_centers = _cdf(reference[:, :, c])
        # For each source pixel value, find the closest reference value that
        # has the same cumulative probability.
        lut = np.interp(src_cdf, ref_cdf, ref_centers).astype(np.float32)
        # Clamp then apply
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        result[:, :, c] = lut[source[:, :, c]]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def get_augmentation(image_size: int) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.2),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class IlluminationDataset(Dataset):
    """
    Returns:
        normal  : Tensor[-1,1] shape (3, H, W)
        target  : Tensor[-1,1] shape (3, H, W)   (over- or under-exposed)
        ev      : float scalar  (positive = over, negative = under)
        label   : int  0=over, 1=under

    domain = "over" | "under" | "both"
      "both" randomly draws from over or under on each __getitem__ call.

    use_pseudo_pairs:
        If True, applies histogram specification so the target is a
        photometrically warped version of the *same* Normal frame.
        This produces geometry-consistent pseudo-pairs and is recommended
        when the datasets are unpaired.
    """

    def __init__(
        self,
        dir_normal: str,
        dir_over: str,
        dir_under: str,
        image_size: int,
        domain: str = "both",
        ev_over_range: Tuple[float, float] = (1.5, 3.0),
        ev_under_range: Tuple[float, float] = (-3.0, -1.5),
        use_pseudo_pairs: bool = True,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.image_size       = image_size
        self.domain           = domain
        self.ev_over_range    = ev_over_range
        self.ev_under_range   = ev_under_range
        self.use_pseudo_pairs = use_pseudo_pairs
        self.augment          = augment

        self.normal_paths = list_images(dir_normal)
        self.over_paths   = list_images(dir_over)
        self.under_paths  = list_images(dir_under)

        if len(self.normal_paths) == 0:
            raise FileNotFoundError(
                f"No images found in Normal directory: {dir_normal!r}. "
                "Please add images or point to the correct path."
            )

        # Warn if target domains are empty (synthetic pseudo-pairs still work)
        if domain in ("over", "both") and len(self.over_paths) == 0:
            print("[Dataset] WARNING: No overexposed images found. "
                  "Pseudo-pairs only for over-domain.")
        if domain in ("under", "both") and len(self.under_paths) == 0:
            print("[Dataset] WARNING: No underexposed images found. "
                  "Pseudo-pairs only for under-domain.")

        self.aug_transform = get_augmentation(image_size) if augment else None
        self._rng = random.Random(seed)

        print(f"[Dataset] Normal={len(self.normal_paths)} | "
              f"Over={len(self.over_paths)} | Under={len(self.under_paths)} | "
              f"domain={domain} | pseudo_pairs={use_pseudo_pairs}")

    def __len__(self) -> int:
        return len(self.normal_paths)

    def _load_pil(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _apply_augmentation(
        self, normal: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply the same random augmentation to both images."""
        # Stack → apply same transform → unstack
        seed = self._rng.randint(0, 2 ** 32)
        random.seed(seed)
        torch.manual_seed(seed)
        normal = self.aug_transform(normal)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.aug_transform(target)
        return normal, target

    def __getitem__(self, idx: int) -> dict:
        normal_pil = self._load_pil(self.normal_paths[idx])
        normal_pil = normal_pil.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )

        # Choose domain for this sample
        if self.domain == "over":
            is_over = True
        elif self.domain == "under":
            is_over = False
        else:  # "both"
            is_over = self._rng.random() < 0.5

        label = 0 if is_over else 1
        ev_range = self.ev_over_range if is_over else self.ev_under_range
        ev = self._rng.uniform(*ev_range)

        # ── Build target image ──────────────────────────────────────────────
        ref_paths = self.over_paths if is_over else self.under_paths

        if self.use_pseudo_pairs or len(ref_paths) == 0:
            # Histogram specification: produce geometry-consistent pseudo-pair
            if len(ref_paths) > 0:
                ref_path = self._rng.choice(ref_paths)
                ref_pil  = self._load_pil(ref_path).resize(
                    (self.image_size, self.image_size), Image.LANCZOS
                )
                normal_np = np.array(normal_pil)
                ref_np    = np.array(ref_pil)
                target_np = histogram_specification(normal_np, ref_np)
                target_pil = Image.fromarray(target_np)
            else:
                # No reference frames available at all — apply parametric
                # gamma simulation as a physics-informed fallback.
                target_pil = _parametric_exposure(normal_pil, ev)
        else:
            # Direct sample from the target domain (unpaired but realistic)
            ref_path  = self._rng.choice(ref_paths)
            target_pil = self._load_pil(ref_path).resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )

        # ── Augmentation (same transform on both) ──────────────────────────
        if self.aug_transform is not None:
            normal_pil, target_pil = self._apply_augmentation(
                normal_pil, target_pil
            )

        normal_t = pil_to_tensor(normal_pil, self.image_size)
        target_t = pil_to_tensor(target_pil, self.image_size)

        return {
            "normal": normal_t,         # (3, H, W) in [-1, 1]
            "target": target_t,         # (3, H, W) in [-1, 1]
            "ev":     torch.tensor(ev, dtype=torch.float32),
            "label":  torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Physics-informed parametric exposure simulation (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _parametric_exposure(pil_img: Image.Image, ev: float) -> Image.Image:
    """
    Simulate exposure shift via gamma + sensor clipping.

    For over-exposure (ev > 0): brighten + hard-clip highlights (sensor
    saturation).
    For under-exposure (ev < 0): darken + add Gaussian noise in shadows
    (shot/read noise simulation).
    """
    arr = np.array(pil_img).astype(np.float32) / 255.0   # [0, 1]

    # Exposure scaling:  pixel * 2^EV
    arr = arr * (2.0 ** ev)

    if ev > 0:
        # Sensor saturation: hard clip at 1.0
        arr = np.clip(arr, 0.0, 1.0)
    else:
        # Shadow noise: scale-dependent Gaussian (shot noise model)
        sigma = 0.01 * (2.0 ** (-ev))   # more noise for stronger underexposure
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        arr   = np.clip(arr + noise, 0.0, 1.0)

    arr_uint8 = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)

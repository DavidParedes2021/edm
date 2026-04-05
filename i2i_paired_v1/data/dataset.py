"""
data/dataset.py
Paired illumination dataset loader.

Each sample yields:
  normal     : Tensor [3, H, W] in [-1, 1]
  artifact   : Tensor [3, H, W] in [-1, 1]  (over- or under-exposed)
  exposure   : float scalar in [-1, +1]
               -1 = max underexposure, +1 = max overexposure
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_image_pairs(artifact_dir: str, normal_dir: str) -> List[Tuple[str, str]]:
    """
    Returns sorted list of (artifact_path, normal_path) pairs.
    Matches by filename – both directories must have identical filenames.
    """
    artifact_dir = Path(artifact_dir)
    normal_dir   = Path(normal_dir)

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {artifact_dir}")
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal dir not found: {normal_dir}")

    artifact_files = sorted([
        p for p in artifact_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ])

    pairs = []
    for art_path in artifact_files:
        norm_path = normal_dir / art_path.name
        if norm_path.exists():
            pairs.append((str(art_path), str(norm_path)))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No paired images found between:\n  {artifact_dir}\n  {normal_dir}\n"
            "Ensure both directories contain files with identical filenames."
        )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Single-mode dataset (over or under)
# ──────────────────────────────────────────────────────────────────────────────

class IlluminationPairDataset(Dataset):
    """
    Dataset for one exposure mode (over or under).

    Args:
        artifact_dir : directory with over/under-exposed frames
        normal_dir   : directory with paired normal frames
        image_size   : spatial resolution (square crop)
        exposure_sign: +1.0 for overexposed, -1.0 for underexposed
        split        : "train" | "val" | "test"
        exposure_range: tuple (min, max) fraction of exposure label used for
                        augmentation (e.g., (0.5, 1.0)).  Only applied during train.
    """

    def __init__(
        self,
        artifact_dir: str,
        normal_dir: str,
        image_size: int,
        exposure_sign: float,
        split: str = "train",
        exposure_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.pairs          = _collect_image_pairs(artifact_dir, normal_dir)
        self.image_size     = image_size
        self.exposure_sign  = exposure_sign          # ±1
        self.split          = split
        self.exposure_range = exposure_range
        self.is_train       = split == "train"

        # Spatial transforms applied identically to both images
        self.resize = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.to_tensor = T.ToTensor()          # [0,1]
        self.normalize = T.Normalize([0.5]*3, [0.5]*3)  # → [-1,1]

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgb(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _paired_augment(
        self, normal: Image.Image, artifact: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply identical spatial augmentation to both images."""
        # Random horizontal flip
        if random.random() > 0.5:
            normal   = TF.hflip(normal)
            artifact = TF.hflip(artifact)

        # Random vertical flip (less common but useful)
        if random.random() > 0.8:
            normal   = TF.vflip(normal)
            artifact = TF.vflip(artifact)

        # Random crop: first resize to slightly larger, then crop
        scale = random.uniform(1.0, 1.15)
        larger = int(self.image_size * scale)
        normal   = TF.resize(normal,   larger, interpolation=TF.InterpolationMode.BICUBIC)
        artifact = TF.resize(artifact, larger, interpolation=TF.InterpolationMode.BICUBIC)

        i, j, h, w = T.RandomCrop.get_params(
            normal, output_size=(self.image_size, self.image_size)
        )
        normal   = TF.crop(normal,   i, j, h, w)
        artifact = TF.crop(artifact, i, j, h, w)

        return normal, artifact

    def _to_tensor_norm(self, img: Image.Image) -> torch.Tensor:
        t = self.to_tensor(img)           # [3,H,W] in [0,1]
        t = self.normalize(t)             # → [-1,1]
        return t

    def __getitem__(self, idx: int) -> dict:
        art_path, norm_path = self.pairs[idx]

        artifact_img = self._load_rgb(art_path)
        normal_img   = self._load_rgb(norm_path)

        # Always resize to base size first
        artifact_img = self.resize(artifact_img)
        normal_img   = self.resize(normal_img)

        # Training-only spatial augmentation
        if self.is_train:
            normal_img, artifact_img = self._paired_augment(normal_img, artifact_img)

        # Exposure label augmentation:
        # Instead of always using full ±1, randomly sample an exposure
        # strength.  This forces the model to generalise over exposure levels.
        if self.is_train:
            strength = random.uniform(*self.exposure_range)
        else:
            strength = 1.0

        exposure_label = float(self.exposure_sign * strength)  # in [-1, +1]

        # If strength < 1.0 we linearly blend artifact toward normal in pixel space
        # so the label and the image are consistent.
        if self.is_train and strength < 1.0:
            art_arr  = np.array(artifact_img, dtype=np.float32)
            norm_arr = np.array(normal_img,   dtype=np.float32)
            blended  = norm_arr + strength * (art_arr - norm_arr)
            blended  = np.clip(blended, 0, 255).astype(np.uint8)
            artifact_img = Image.fromarray(blended)

        normal_t   = self._to_tensor_norm(normal_img)
        artifact_t = self._to_tensor_norm(artifact_img)

        return {
            "normal":   normal_t,                              # [3,H,W]
            "artifact": artifact_t,                            # [3,H,W]
            "exposure": torch.tensor(exposure_label, dtype=torch.float32),  # scalar
            "art_path": art_path,
            "norm_path": norm_path,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Combined over + under dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, split: str = "train"):
    """
    Build DataLoaders for train/val/test splits.

    Args:
        cfg  : top-level config dict (from yaml)
        split: "train" | "validation" | "test"

    Returns:
        DataLoader combining over- and under-exposed pairs.
    """
    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]
    image_size = data_cfg["image_size"]

    # ── Under-exposed dataset ──────────────────────────────────────────
    under_art_dir  = os.path.join(data_cfg["underexposed_root"], split, "underexposed")
    under_norm_dir = os.path.join(data_cfg["underexposed_root"], split, "normal_frames")

    # ── Over-exposed dataset ───────────────────────────────────────────
    over_art_dir   = os.path.join(data_cfg["overexposed_root"], split, "overexposed")
    over_norm_dir  = os.path.join(data_cfg["overexposed_root"], split, "normal_frames")

    datasets = []

    if os.path.exists(under_art_dir) and os.path.exists(under_norm_dir):
        datasets.append(
            IlluminationPairDataset(
                artifact_dir   = under_art_dir,
                normal_dir     = under_norm_dir,
                image_size     = image_size,
                exposure_sign  = -1.0,
                split          = split,
            )
        )
    else:
        print(f"[WARN] Under-exposed {split} dirs not found, skipping.")

    if os.path.exists(over_art_dir) and os.path.exists(over_norm_dir):
        datasets.append(
            IlluminationPairDataset(
                artifact_dir  = over_art_dir,
                normal_dir    = over_norm_dir,
                image_size    = image_size,
                exposure_sign = +1.0,
                split         = split,
            )
        )
    else:
        print(f"[WARN] Over-exposed {split} dirs not found, skipping.")

    if len(datasets) == 0:
        raise RuntimeError(f"No datasets found for split='{split}'")

    combined = ConcatDataset(datasets)

    is_train   = split == "train"
    batch_size = train_cfg["batch_size"] if is_train else max(1, train_cfg["batch_size"] // 2)
    num_workers = data_cfg.get("num_workers", 0)

    loader = DataLoader(
        combined,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = is_train,
    )
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.config_utils import load_config

    cfg = load_config("configs/debug.yaml")
    loader = build_dataloaders(cfg, split="train")
    batch = next(iter(loader))
    print("normal  :", batch["normal"].shape,   batch["normal"].min().item(),   batch["normal"].max().item())
    print("artifact:", batch["artifact"].shape, batch["artifact"].min().item(), batch["artifact"].max().item())
    print("exposure:", batch["exposure"])

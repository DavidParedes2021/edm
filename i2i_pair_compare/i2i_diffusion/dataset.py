"""Paired dataset for I2I exposure diffusion.

Reads the four directories produced by `generate_pairs.py`:
    pairs_root/
        normal/   <stem>.npy   (H, W, 3)  full LAB, float32, L in [0, 100]
        depth/    <stem>.npy   (H, W)     float16, [0, 1], 1.0 = nearest
        over/     <stem>[_v*].npy  (H, W) float32, augmented L in [0, 100]
        under/    <stem>[_v*].npy  (H, W) float32, augmented L in [0, 100]

For each item we sample one mode (over=0 or under=1) and return:
    cond_L    (1, H, W) in [-1, 1]   normal luminance
    depth     (1, H, W) in [-1, 1]
    target_L  (1, H, W) in [-1, 1]   chosen mode L
    mode      scalar long {0, 1}

Train/val split is done on the *base stem* — all variants of the same source
image fall in the same split, preventing leakage.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _strip_variant(stem: str) -> str:
    """`abc_v0` -> `abc`. `abc` -> `abc`."""
    if "_v" in stem:
        head, tail = stem.rsplit("_v", 1)
        if tail.isdigit():
            return head
    return stem


class PairDataset(Dataset):
    def __init__(
        self,
        pairs_root: str | Path,
        resolution: int = 256,
        flip_prob: float = 0.5,
        split: str = "all",          # 'all' | 'train' | 'val'
        val_fraction: float = 0.0,
        split_seed: int = 1234,
        augment: bool = True,        # disable for validation: center crop, no flip, deterministic mode
    ) -> None:
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be 'all'/'train'/'val', got {split}")
        root = Path(pairs_root)
        self.normal_dir = root / "normal"
        self.depth_dir = root / "depth"
        self.over_dir = root / "over"
        self.under_dir = root / "under"
        for d in (self.normal_dir, self.depth_dir, self.over_dir, self.under_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Missing directory: {d}")

        normal_stems = {p.stem for p in self.normal_dir.glob("*.npy")}
        depth_stems = {p.stem for p in self.depth_dir.glob("*.npy")}
        over_files = list(self.over_dir.glob("*.npy"))
        under_files = list(self.under_dir.glob("*.npy"))

        over_by_stem = {p.stem: p for p in over_files}
        under_by_stem = {p.stem: p for p in under_files}

        items: List[Tuple[str, Path, Path]] = []
        for ostem, opath in over_by_stem.items():
            base = _strip_variant(ostem)
            if base not in normal_stems or base not in depth_stems:
                continue
            if ostem in under_by_stem:
                items.append((base, opath, under_by_stem[ostem]))
            elif base in under_by_stem:
                items.append((base, opath, under_by_stem[base]))
        if not items:
            raise RuntimeError(
                f"No matching (normal, depth, over, under) tuples found under {root}"
            )

        # ── Train/val split on base stem ──────────────────────────────────
        all_bases = sorted({b for b, _, _ in items})
        if split != "all" and val_fraction > 0.0:
            rng = random.Random(split_seed)
            shuffled = list(all_bases)
            rng.shuffle(shuffled)
            n_val = int(round(len(shuffled) * float(val_fraction)))
            n_val = max(1, n_val) if val_fraction > 0 and len(shuffled) > 1 else n_val
            val_bases = set(shuffled[:n_val])
            train_bases = set(shuffled[n_val:])
            keep = val_bases if split == "val" else train_bases
            items = [it for it in items if it[0] in keep]

        self.items = items
        self.resolution = int(resolution)
        self.flip_prob = float(flip_prob) if augment else 0.0
        self.augment = bool(augment)
        self.split = split

    def __len__(self) -> int:
        return len(self.items)

    def _crop_indices(self, H: int, W: int, idx: int) -> Tuple[int, int]:
        res = self.resolution
        if H < res or W < res:
            raise ValueError(f"Image {H}x{W} smaller than resolution {res}")
        if self.augment:
            return random.randint(0, H - res), random.randint(0, W - res)
        # center crop for val determinism
        return (H - res) // 2, (W - res) // 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base, over_path, under_path = self.items[idx]
        normal_lab = np.load(self.normal_dir / f"{base}.npy")  # (H, W, 3) float32
        depth = np.load(self.depth_dir / f"{base}.npy").astype(np.float32)  # (H, W)
        over_L = np.load(over_path).astype(np.float32)
        under_L = np.load(under_path).astype(np.float32)

        L_normal = normal_lab[..., 0].astype(np.float32)
        H, W = L_normal.shape
        if not (depth.shape == (H, W) == over_L.shape == under_L.shape):
            raise ValueError(
                f"Shape mismatch for {base}: "
                f"normalL={L_normal.shape}, depth={depth.shape}, "
                f"over={over_L.shape}, under={under_L.shape}"
            )

        top, left = self._crop_indices(H, W, idx)
        res = self.resolution
        sl = (slice(top, top + res), slice(left, left + res))
        L_normal = L_normal[sl]
        depth_c = depth[sl]
        over_c = over_L[sl]
        under_c = under_L[sl]

        if self.augment and random.random() < self.flip_prob:
            L_normal = np.ascontiguousarray(L_normal[:, ::-1])
            depth_c = np.ascontiguousarray(depth_c[:, ::-1])
            over_c = np.ascontiguousarray(over_c[:, ::-1])
            under_c = np.ascontiguousarray(under_c[:, ::-1])

        # mode 0=over, 1=under
        # In val mode use both deterministically: alternate by item index so
        # the val loss is a stable mix of both modes.
        if self.augment:
            mode = random.randint(0, 1)
        else:
            mode = idx % 2
        target_L = over_c if mode == 0 else under_c

        L_normal_n = (L_normal / 50.0 - 1.0).astype(np.float32)
        depth_n = (depth_c * 2.0 - 1.0).astype(np.float32)
        target_n = (target_L / 50.0 - 1.0).astype(np.float32)

        return {
            "cond_L": torch.from_numpy(L_normal_n).unsqueeze(0),
            "depth": torch.from_numpy(depth_n).unsqueeze(0),
            "target_L": torch.from_numpy(target_n).unsqueeze(0),
            "mode": torch.tensor(mode, dtype=torch.long),
        }

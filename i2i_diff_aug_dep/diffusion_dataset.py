"""
diffusion_dataset.py — Paired L-channel dataset for conditional diffusion
with depth conditioning.

Changes vs. previous version:
  - PairedLuminanceDataset now loads depth maps and returns them alongside
    source/target L as a 3rd channel for the UNet input.
  - Joint spatial augmentation applied IDENTICALLY to source_L, target_L,
    and depth. Without this, the model sees mis-aligned supervision.
  - NormalInferenceDataset now returns depth too, either loaded from disk
    or — if absent — computed on-the-fly.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from exposure_augment import rgb_to_lab


# --------------------------------------------------------------------------- #
# Training dataset: paired (normal L, depth) → target L
# --------------------------------------------------------------------------- #

class PairedLuminanceDataset(Dataset):
    """Each item returns:
        source_L : (1, H, W) float32 in [-1, 1]
        target_L : (1, H, W) float32 in [-1, 1]
        depth    : (1, H, W) float32 in [-1, 1]  (rescaled from [0,1])
        label    : int  (0=overexposed, 1=underexposed)
    """

    def __init__(
        self,
        pairs_dir: str,
        image_size: int = 256,
        augment: bool = True,
        gamma_jitter: float = 0.1,        # ± range for random gamma on L
        depth_noise_sigma: float = 0.02,  # Gaussian noise on depth (robustness)
    ):
        self.image_size = image_size
        self.augment = augment
        self.gamma_jitter = gamma_jitter
        self.depth_noise_sigma = depth_noise_sigma

        pairs = Path(pairs_dir)
        normal_dir = pairs / "normal"
        depth_dir = pairs / "depth"
        over_dir = pairs / "overexposed"
        under_dir = pairs / "underexposed"

        if not depth_dir.is_dir():
            raise RuntimeError(
                f"Depth directory missing: {depth_dir}. "
                f"Run depth_estimator.py and regenerate pairs."
            )

        self.samples = []  # (normal_lab_path, depth_path, target_L_path, label)
        for npy in sorted(normal_dir.glob("*.npy")):
            stem = npy.stem
            dpath = depth_dir / f"{stem}.npy"
            if not dpath.exists():
                continue
            for t in sorted(over_dir.glob(f"{stem}*.npy")):
                self.samples.append((str(npy), str(dpath), str(t), 0))
            for t in sorted(under_dir.glob(f"{stem}*.npy")):
                self.samples.append((str(npy), str(dpath), str(t), 1))

        if not self.samples:
            raise RuntimeError(
                f"No paired samples in {pairs_dir}. Run generate_pairs.py first."
            )

        # balance domains
        over = [s for s in self.samples if s[3] == 0]
        under = [s for s in self.samples if s[3] == 1]
        n_o, n_u = len(over), len(under)
        majority = max(n_o, n_u)
        if n_o < majority and n_o > 0:
            self.samples.extend(random.choices(over, k=majority - n_o))
        elif n_u < majority and n_u > 0:
            self.samples.extend(random.choices(under, k=majority - n_u))
        random.shuffle(self.samples)

        print(f"[PairedDataset] {len(self.samples)} pairs "
              f"(over={n_o}, under={n_u}, balanced to {len(self.samples)})")

    # ──────────────────────────────────────────────────────────────────── #
    @staticmethod
    def _normalise_L(L):
        return (L / 50.0) - 1.0

    @staticmethod
    def _normalise_depth(d):
        # [0, 1] → [-1, 1]  to match the UNet input scale
        return d * 2.0 - 1.0

    @staticmethod
    def _resize_2d(arr, size):
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────── #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        normal_path, depth_path, target_path, label = self.samples[idx]

        normal_lab = np.load(normal_path).astype(np.float32)  # (H, W, 3)
        source_L = normal_lab[..., 0]

        target_L = np.load(target_path).astype(np.float32)
        depth = np.load(depth_path).astype(np.float32)

        # resize all to model resolution
        source_L = self._resize_2d(source_L, self.image_size)
        target_L = self._resize_2d(target_L, self.image_size)
        depth = self._resize_2d(depth, self.image_size)
        depth = np.clip(depth, 0.0, 1.0)  # preserve domain

        # ── joint spatial augmentation (SAME transform on all three) ──
        if self.augment:
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=1).copy()
                target_L = np.flip(target_L, axis=1).copy()
                depth    = np.flip(depth,    axis=1).copy()
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=0).copy()
                target_L = np.flip(target_L, axis=0).copy()
                depth    = np.flip(depth,    axis=0).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                source_L = np.rot90(source_L, k).copy()
                target_L = np.rot90(target_L, k).copy()
                depth    = np.rot90(depth,    k).copy()

            # gamma jitter on L channels — simulate light-source intensity variation.
            # Apply IDENTICALLY to source and target so the residual is preserved.
            if self.gamma_jitter > 0:
                g = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
                # map to [0,1], exponentiate, map back
                s01 = np.clip(source_L / 100.0, 0.0, 1.0)
                t01 = np.clip(target_L / 100.0, 0.0, 1.0)
                source_L = (np.power(s01, g) * 100.0).astype(np.float32)
                target_L = (np.power(t01, g) * 100.0).astype(np.float32)

            # depth noise — robustness to depth-estimator errors
            if self.depth_noise_sigma > 0:
                depth = depth + np.random.normal(0.0, self.depth_noise_sigma, depth.shape).astype(np.float32)
                depth = np.clip(depth, 0.0, 1.0)

        # normalise to [-1, 1]
        source_L = self._normalise_L(source_L)
        target_L = self._normalise_L(target_L)
        depth = self._normalise_depth(depth)

        source_tensor = torch.from_numpy(source_L).unsqueeze(0)
        target_tensor = torch.from_numpy(target_L).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return source_tensor, target_tensor, depth_tensor, label


# --------------------------------------------------------------------------- #
# Inference dataset: loads normal RGB + depth at original and model resolution
# --------------------------------------------------------------------------- #

class NormalInferenceDataset(Dataset):
    """Returns:
        source_L   : (1, H_model, W_model)  normalised L, model resolution
        depth_s    : (1, H_model, W_model)  normalised depth, model resolution
        AB         : (H_orig, W_orig, 2)    original-resolution chroma
        L_orig     : (H_orig, W_orig)       original-resolution L
        depth_full : (H_orig, W_orig)       original-resolution depth  (for chroma fix)
        path       : str
        orig_hw    : (int, int)             original height, width
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        normal_dir: str,
        image_size: int = 256,
        depth_dir: str = None,        # if None, depth is computed on-the-fly
        depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ):
        self.image_size = image_size
        folder = Path(normal_dir)
        self.paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in self.EXTENSIONS
        )
        print(f"[InferenceDataset] {len(self.paths)} images from {normal_dir}")

        self.depth_dir = Path(depth_dir) if depth_dir else None
        self._depth_pipe = None
        self._depth_model_id = depth_model_id

    def _ensure_depth_pipe(self):
        if self._depth_pipe is None:
            from transformers import pipeline
            import torch as _t
            dev = 0 if _t.cuda.is_available() else -1
            print(f"[InferenceDataset] lazy-loading {self._depth_model_id}")
            self._depth_pipe = pipeline(
                task="depth-estimation", model=self._depth_model_id, device=dev
            )

    def _load_or_compute_depth(self, path: Path, img_pil: Image.Image) -> np.ndarray:
        # 1. try disk cache
        if self.depth_dir is not None:
            candidate = self.depth_dir / f"{path.stem}.npy"
            if candidate.exists():
                d = np.load(str(candidate)).astype(np.float32)
                return np.clip(d, 0.0, 1.0)
        # 2. compute inline
        self._ensure_depth_pipe()
        out = self._depth_pipe(img_pil)
        import torch as _t
        dt = out["predicted_depth"]
        if dt.ndim == 2:
            dt = dt.unsqueeze(0).unsqueeze(0)
        elif dt.ndim == 3:
            dt = dt.unsqueeze(0)
        W, H = img_pil.size
        dt = _t.nn.functional.interpolate(dt, size=(H, W), mode="bicubic", align_corners=False)
        d = dt.squeeze().cpu().numpy().astype(np.float32)
        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-6:
            return np.full_like(d, 0.5, dtype=np.float32)
        return ((d - mn) / (mx - mn)).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        lab = rgb_to_lab(img_np)

        L_orig = lab[..., 0].copy()
        AB = lab[..., 1:].copy()

        depth_full = self._load_or_compute_depth(path, img_pil)  # (H, W)

        # ── resized model-input tensors ──
        L_s = np.array(
            Image.fromarray(L_orig.astype(np.float32), mode="F").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        L_s = (L_s / 50.0) - 1.0

        D_s = np.array(
            Image.fromarray(depth_full.astype(np.float32), mode="F").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        D_s = np.clip(D_s, 0.0, 1.0) * 2.0 - 1.0  # [-1, 1]

        source_tensor = torch.from_numpy(L_s).unsqueeze(0)
        depth_tensor = torch.from_numpy(D_s).unsqueeze(0)

        return (
            source_tensor,
            depth_tensor,
            AB,
            L_orig,
            depth_full,
            str(path),
            img_np.shape[:2],
        )

"""Datasets for L-channel inpainting diffusion.

ArtifactInpaintingDataset
    Yields (L*, mask, depth, valid) tensors from real over/under-exposed frames.
    The mask is computed FROM THE ARTIFACT FRAME so that diffusion learns the
    distribution of the artifact L pattern conditioned on its surrounding
    context. This is the correct training target for the inference task.

NormalSampleDataset
    Yields the same fields from real normal frames (mask is derived from the
    photometric depth proxy). Used for periodic visualization during training
    and for final batch inference.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from . import color as colorm
from . import depth as depthm


def _blur2d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Reflect-padded Gaussian blur. sigma <= 0 returns the input unchanged."""
    if sigma is None or float(sigma) <= 0.0:
        return x.astype(np.float32, copy=True)
    return gaussian_filter(x, sigma=float(sigma), mode="reflect").astype(np.float32)


_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def _list_images(d: str):
    paths = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(d, ext)))
        paths.extend(glob.glob(os.path.join(d, ext.upper())))
    return sorted(set(paths))


def _load_rgb(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    # np.array (not np.asarray) forces a writable copy -- avoids
    # "non-writable tensor" warnings when feeding to torch.from_numpy later.
    return np.array(img)


class ArtifactInpaintingDataset(Dataset):
    """Real artifact frames -> (L, L_lf, mask, depth, valid) for inpainting diffusion.

    `L_lf` is the Gaussian-blurred (low-frequency) component of the L channel.
    Trainers may use `L_lf` as the diffusion target (`predict_lf_target=true`)
    to focus model capacity on the illumination shift instead of texture.
    """

    def __init__(self, img_dir: str, image_size: int, artifact: str,
                 mask_cfg: dict, blur_sigma_lf: float = 0.0):
        if artifact not in ("overexposure", "underexposure"):
            raise ValueError(f"unknown artifact: {artifact}")
        self.paths = _list_images(img_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {img_dir}")
        self.image_size = int(image_size)
        self.artifact = artifact
        self.mask_cfg = mask_cfg
        self.blur_sigma_lf = float(blur_sigma_lf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        rgb = _load_rgb(self.paths[idx], self.image_size)
        # Random horizontal flip for mild augmentation.
        if np.random.rand() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
        L, _ab = colorm.rgb_to_lab(rgb)
        valid = depthm.valid_frame_mask(rgb)
        d_proxy = depthm.depth_proxy(rgb)
        if self.artifact == "overexposure":
            mask = depthm.overexposure_mask_from_artifact(
                L, valid,
                self.mask_cfg["cluster_fraction"],
                self.mask_cfg["dilation_iters"],
                self.mask_cfg["blur_sigma"],
                self.mask_cfg["min_component_area_frac"],
            )
        else:
            mask = depthm.underexposure_mask_from_artifact(
                L, valid,
                self.mask_cfg["cluster_fraction"],
                self.mask_cfg["dilation_iters"],
                self.mask_cfg["blur_sigma"],
                self.mask_cfg["min_component_area_frac"],
            )
        L_norm    = colorm.normalize_L(L).astype(np.float32)
        L_norm_lf = _blur2d(L_norm, self.blur_sigma_lf)
        return {
            "L":     torch.from_numpy(L_norm).unsqueeze(0),                       # (1,H,W)
            "L_lf":  torch.from_numpy(L_norm_lf).unsqueeze(0),                    # (1,H,W)
            "mask":  torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),      # (1,H,W)
            "depth": torch.from_numpy(d_proxy.astype(np.float32)).unsqueeze(0),   # (1,H,W)
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),     # (1,H,W)
        }


class NormalSampleDataset(Dataset):
    """Normal frames with depth-derived masks for inference / periodic samples."""

    def __init__(self, img_dir: str, image_size: int, artifact: str,
                 mask_cfg: dict, limit=None, blur_sigma_lf: float = 0.0):
        if artifact not in ("overexposure", "underexposure"):
            raise ValueError(f"unknown artifact: {artifact}")
        self.paths = _list_images(img_dir)
        if limit is not None:
            self.paths = self.paths[:int(limit)]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {img_dir}")
        self.image_size = int(image_size)
        self.artifact = artifact
        self.mask_cfg = mask_cfg
        self.blur_sigma_lf = float(blur_sigma_lf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        rgb = _load_rgb(self.paths[idx], self.image_size)
        L, ab = colorm.rgb_to_lab(rgb)
        valid = depthm.valid_frame_mask(rgb)
        d_proxy = depthm.depth_proxy(rgb)
        if self.artifact == "overexposure":
            mask = depthm.overexposure_mask_from_normal(
                rgb,
                self.mask_cfg["cluster_fraction"],
                self.mask_cfg["dilation_iters"],
                self.mask_cfg["blur_sigma"],
                self.mask_cfg["min_component_area_frac"],
            )
        else:
            mask = depthm.underexposure_mask_from_normal(
                rgb,
                self.mask_cfg["cluster_fraction"],
                self.mask_cfg["dilation_iters"],
                self.mask_cfg["blur_sigma"],
                self.mask_cfg["min_component_area_frac"],
            )
        L_norm    = colorm.normalize_L(L).astype(np.float32)
        L_norm_lf = _blur2d(L_norm, self.blur_sigma_lf)
        return {
            "L":     torch.from_numpy(L_norm).unsqueeze(0),
            "L_lf":  torch.from_numpy(L_norm_lf).unsqueeze(0),
            "mask":  torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "depth": torch.from_numpy(d_proxy.astype(np.float32)).unsqueeze(0),
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
            "rgb":   torch.from_numpy(rgb),                                # uint8 (H,W,3)
            "ab":    torch.from_numpy(ab.astype(np.float32)),              # (H,W,2)
            "path":  self.paths[idx],
        }

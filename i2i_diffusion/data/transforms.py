"""
data/transforms.py
------------------
Augmentation pipelines and histogram-matching warm-start utility.

All colour transforms must be identical across the normal/over/under
images within a single batch *pair* to avoid geometric mismatch — here
we apply them independently because the dataset is unpaired, so geometric
coherence is enforced only through the structural (edge-map) conditioning.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ── edge extraction ───────────────────────────────────────────────────────────

def _pil_to_edge(img: Image.Image) -> Image.Image:
    """
    Returns a 1-channel PIL image containing the normalised Sobel gradient
    magnitude.  We use PIL's FIND_EDGES filter (Sobel-like) which is
    available in Pillow 9.x without any C extension dependency.
    """
    gray = img.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    return edge  # still PIL "L" mode


# ── histogram matching ────────────────────────────────────────────────────────

def histogram_match_pil(
    source: Image.Image, reference: Image.Image
) -> Image.Image:
    """
    Adjust `reference`'s channel histograms to match `source`.
    This gives the model a luminance-matched starting point so it only
    needs to learn fine-grained exposure physics, not coarse tone shifts.

    Works channel-by-channel in RGB.  Pure numpy — no skimage needed.
    """
    src  = np.array(source, dtype=np.float32)
    ref  = np.array(reference, dtype=np.float32)
    out  = np.empty_like(ref)

    for c in range(3):
        s = src[..., c].ravel()
        r = ref[..., c].ravel()

        # CDFs
        s_vals, s_counts = np.unique(s, return_counts=True)
        r_vals, r_counts = np.unique(r, return_counts=True)

        s_cdf = np.cumsum(s_counts).astype(np.float64)
        s_cdf /= s_cdf[-1]
        r_cdf = np.cumsum(r_counts).astype(np.float64)
        r_cdf /= r_cdf[-1]

        # map each ref pixel value to the closest source CDF level
        interp = np.interp(r_cdf, s_cdf, s_vals)
        lut    = np.interp(np.arange(256), r_vals, interp).clip(0, 255)

        out[..., c] = lut[ref[..., c].astype(np.uint8)]

    return Image.fromarray(out.astype(np.uint8), mode="RGB")


# ── normalise helpers ─────────────────────────────────────────────────────────

def _to_tensor_minus1_1(img: Image.Image) -> torch.Tensor:
    """PIL RGB → float32 tensor in [-1, 1]."""
    t = TF.to_tensor(img)          # [0, 1]
    return t * 2.0 - 1.0


def _edge_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL L (edge) → float32 tensor [1, H, W] in [-1, 1]."""
    t = TF.to_tensor(img)          # [0, 1], shape [1, H, W]
    return t * 2.0 - 1.0


# ── transform factories ───────────────────────────────────────────────────────

class _EdgeTransform:
    """Resize → crop → extract edges → tensor."""

    def __init__(self, image_size: int, augment: bool) -> None:
        self.image_size = image_size
        self.augment    = augment

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # geometry
        img = TF.resize(img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, self.image_size)
        if self.augment and torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        # edge
        edge = _pil_to_edge(img)
        return _edge_to_tensor(edge)


class _ImageTransform:
    """Resize → crop → optional flip → tensor in [-1,1]."""

    def __init__(self, image_size: int, augment: bool) -> None:
        self.image_size = image_size
        self.augment    = augment

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, self.image_size)
        if self.augment and torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        return _to_tensor_minus1_1(img)


def build_transforms(
    image_size: int,
    augment: bool = True,
    edge: bool = False,
) -> _ImageTransform | _EdgeTransform:
    if edge:
        return _EdgeTransform(image_size, augment)
    return _ImageTransform(image_size, augment)

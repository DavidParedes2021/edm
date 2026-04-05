"""
dataset.py — Dataset utilities for the Illumination Diffusion pipeline.

Supports:
  • Unpaired Normal / Overexposed / Underexposed directories.
  • On-the-fly histogram-specification pseudo-pairing (optional).
  • Physics-informed parametric exposure simulation (improved).
  • Returns (normal_img, target_img, ev_scalar) triplets.

All images are returned as torch.Tensor in [-1, 1] (VAE convention).

Improvements vs original:
  1. Stronger augmentation: vertical flip, rotation, random crop → vital for
     training on very small datasets.
  2. Colour jitter removed from the paired augmentation — it changes the
     relative luminance relationship between normal and target, which
     is exactly the signal the model is learning to produce.
  3. Better overexposure simulation: bloom/glow effect via highlight blending.
  4. Better underexposure simulation: Poisson-Gaussian noise model matching
     real camera sensor behaviour.
  5. Vignetting simulation: common in endoscopy for both exposure types.
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
import torchvision.transforms.functional as TF


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


def _warp_channel(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
    """Apply histogram specification to a single uint8 channel array."""
    src_cdf, _           = _cdf(src_ch)
    ref_cdf, ref_centers = _cdf(ref_ch)
    lut = np.interp(src_cdf, ref_cdf, ref_centers).astype(np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[src_ch]


def histogram_specification(
    source: np.ndarray,    # uint8 HxWxC  (Normal frame)
    reference: np.ndarray  # uint8 HxWxC  (Overexposed/Underexposed sample)
) -> np.ndarray:
    """
    Warp *source*'s luminance to match *reference*'s luminance distribution,
    while preserving the source's original hue and saturation (Cb, Cr).

    ROOT CAUSE FIX: The original code warped R, G, B channels independently.
    Independent per-channel histogram matching destroys inter-channel correlation:
    a red pixel (R=200, G=50, B=50) could become pink (R=220, G=130, B=110)
    because the G and B LUTs happen to map low values upward. This corrupts hue
    in the pseudo-pair target, and the model then learns to reproduce wrong colors.

    Real camera exposure changes primarily affect luminance, NOT hue.
    Fix: convert to YCbCr, warp only the Y channel, keep source Cb and Cr intact.
    This produces targets with correct hue/saturation and only shifted brightness.
    """
    src_pil = Image.fromarray(source)
    ref_pil = Image.fromarray(reference)

    # Work in YCbCr so luminance and chrominance are separable
    src_ycbcr = np.array(src_pil.convert("YCbCr"))   # uint8 (H, W, 3)
    ref_ycbcr = np.array(ref_pil.convert("YCbCr"))

    # Warp only the Y (luminance) channel; Cb and Cr come from source unchanged
    result_ycbcr = src_ycbcr.copy()
    result_ycbcr[:, :, 0] = _warp_channel(src_ycbcr[:, :, 0],
                                           ref_ycbcr[:, :, 0])

    # Convert back to RGB
    result_pil = Image.fromarray(result_ycbcr, mode="YCbCr")
    return np.array(result_pil.convert("RGB"))


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation (spatial-only, applied identically to both images in a pair)
# ──────────────────────────────────────────────────────────────────────────────

class PairedAugmentation:
    """
    Apply the same random spatial transform to both the normal and target images.

    Only spatial transforms are applied — no colour jitter.  Colour jitter
    would alter the *relative* luminance/colour relationship between the normal
    and target, which is exactly the signal the model learns to reproduce.

    Transforms:
      • Random horizontal flip (p=0.5)
      • Random vertical flip (p=0.3) — valid for endoscopy (no natural up/down)
      • Random rotation [-10°, +10°]
      • Random crop with padding (crops 10–15% then resizes back)
    """

    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(
        self,
        normal: Image.Image,
        target: Image.Image,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        rng = random.Random(seed)

        # Apply identical random ops to both images
        # 1. Horizontal flip
        if rng.random() < 0.5:
            normal = TF.hflip(normal)
            target = TF.hflip(target)

        # 2. Vertical flip (valid for endoscopy — no natural orientation)
        if rng.random() < 0.3:
            normal = TF.vflip(normal)
            target = TF.vflip(target)

        # 3. Random rotation [-10, +10] degrees
        angle = rng.uniform(-10.0, 10.0)
        normal = TF.rotate(normal, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        target = TF.rotate(target, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)

        # 4. Random crop: crop to 85–95% of the image, then resize back
        crop_fraction = rng.uniform(0.85, 0.95)
        crop_size = int(self.image_size * crop_fraction)
        # Random crop position
        max_offset = self.image_size - crop_size
        top  = rng.randint(0, max_offset)
        left = rng.randint(0, max_offset)
        normal = TF.resized_crop(normal, top, left, crop_size, crop_size,
                                 (self.image_size, self.image_size),
                                 interpolation=TF.InterpolationMode.LANCZOS)
        target = TF.resized_crop(target, top, left, crop_size, crop_size,
                                 (self.image_size, self.image_size),
                                 interpolation=TF.InterpolationMode.LANCZOS)

        return normal, target


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
        Produces geometry-consistent pseudo-pairs, recommended for unpaired data.
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

        if domain in ("over", "both") and len(self.over_paths) == 0:
            print("[Dataset] WARNING: No overexposed images found. "
                  "Pseudo-pairs only for over-domain.")
        if domain in ("under", "both") and len(self.under_paths) == 0:
            print("[Dataset] WARNING: No underexposed images found. "
                  "Pseudo-pairs only for under-domain.")

        self.paired_aug = PairedAugmentation(image_size) if augment else None
        self._rng = random.Random(seed)

        print(f"[Dataset] Normal={len(self.normal_paths)} | "
              f"Over={len(self.over_paths)} | Under={len(self.under_paths)} | "
              f"domain={domain} | pseudo_pairs={use_pseudo_pairs} | "
              f"augment={augment}")

    def __len__(self) -> int:
        """
        Return an epoch length that gives ~100% coverage of both reference pools.

        With a fixed 50/50 domain split and epoch_len = len(normal_paths), only
        30% of a large reference pool might be sampled per epoch if that pool is
        much bigger than the normal count. Setting epoch_len = len(over) + len(under)
        with a proportional domain split ensures every reference image is sampled
        approximately once per epoch, regardless of pool size asymmetry.

        Normal images are accessed round-robin with modulo; augmentation seeds
        vary per access so repeated normal images are not wasted iterations.
        """
        n_over  = len(self.over_paths)
        n_under = len(self.under_paths)
        if self.domain == "over":
            return max(n_over, 200)
        elif self.domain == "under":
            return max(n_under, 200)
        else:
            # "both": cover both reference pools; minimum 200 for smoke tests
            return max(n_over + n_under, 200)

    @property
    def _p_over(self) -> float:
        """
        Proportional probability of sampling the overexposed domain.

        A fixed 50/50 split is only correct when both reference pools are the
        same size. When over=281 and under=817, a 50/50 split means the
        underexposed pool is only ~31% covered per epoch while the overexposed
        pool is ~90% covered — the model trains the two domains at very different
        rates. Proportional sampling gives equal per-image coverage across both
        pools.
        """
        n_over  = len(self.over_paths)
        n_under = len(self.under_paths)
        total   = n_over + n_under
        if total == 0:
            return 0.5
        return n_over / total

    def _load_pil(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        # Normal images are the content source; index wraps round-robin so
        # repeated accesses get different augmentation seeds.
        real_idx   = idx % len(self.normal_paths)
        normal_pil = self._load_pil(self.normal_paths[real_idx])
        normal_pil = normal_pil.resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )

        # Choose domain — proportional to reference pool sizes so both pools
        # receive equal per-image coverage per epoch.
        if self.domain == "over":
            is_over = True
        elif self.domain == "under":
            is_over = False
        else:  # "both" — proportional split
            is_over = self._rng.random() < self._p_over

        label    = 0 if is_over else 1
        ev_range = self.ev_over_range if is_over else self.ev_under_range
        ev       = self._rng.uniform(*ev_range)

        # ── Build target image ──────────────────────────────────────────────
        ref_paths = self.over_paths if is_over else self.under_paths

        if self.use_pseudo_pairs or len(ref_paths) == 0:
            if len(ref_paths) > 0:
                ref_path  = self._rng.choice(ref_paths)
                ref_pil   = self._load_pil(ref_path).resize(
                    (self.image_size, self.image_size), Image.LANCZOS
                )
                normal_np = np.array(normal_pil)
                ref_np    = np.array(ref_pil)
                target_np = histogram_specification(normal_np, ref_np)
                target_pil = Image.fromarray(target_np)
            else:
                # No reference frames — physics-informed simulation (improved)
                target_pil = _parametric_exposure(normal_pil, ev)
        else:
            ref_path   = self._rng.choice(ref_paths)
            target_pil = self._load_pil(ref_path).resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )

        # ── Paired spatial augmentation ────────────────────────────────────
        if self.paired_aug is not None:
            aug_seed = self._rng.randint(0, 2 ** 32 - 1)
            normal_pil, target_pil = self.paired_aug(normal_pil, target_pil,
                                                      seed=aug_seed)

        normal_t = pil_to_tensor(normal_pil, self.image_size)
        target_t = pil_to_tensor(target_pil, self.image_size)

        return {
            "normal": normal_t,
            "target": target_t,
            "ev":     torch.tensor(ev, dtype=torch.float32),
            "label":  torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Physics-informed parametric exposure simulation
# ──────────────────────────────────────────────────────────────────────────────

def _parametric_exposure(pil_img: Image.Image, ev: float) -> Image.Image:
    """
    Simulate camera exposure shift with physically-motivated effects.

    Over-exposure (ev > 0):
      • Scale pixel values by 2^EV (linear sensor response)
      • Hard-clip highlights at 1.0 (sensor saturation / blooming)
      • Add soft bloom glow: highlights bleed into neighbouring pixels
        (approximated by blending a blurred saturated mask back)
      • Slight colour desaturation in overexposed regions (real sensors
        tend towards white in heavily saturated areas)

    Under-exposure (ev < 0):
      • Scale pixel values by 2^EV
      • Add Poisson-Gaussian noise (matches real camera sensor):
          - Poisson noise: variance ∝ signal (shot noise, dark signal limited)
          - Gaussian noise: fixed read noise floor
      • Slight vignetting (stronger in low-light, amplified by dark frame)

    The original code used a simple Gaussian noise model which is only
    valid for read-noise dominated images (very dark scenes).  Real sensors
    follow a Poisson-Gaussian mixed model.
    """
    arr = np.array(pil_img).astype(np.float32) / 255.0   # [0, 1]

    if ev > 0:
        arr = _simulate_overexposure(arr, ev)
    else:
        arr = _simulate_underexposure(arr, ev)

    arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def _simulate_overexposure(arr: np.ndarray, ev: float) -> np.ndarray:
    """
    Overexposure: linear amplification + sensor saturation + bloom glow.
    """
    from scipy.ndimage import gaussian_filter

    # 1. Linear exposure scaling
    bright = arr * (2.0 ** ev)

    # 2. Hard clipping at saturation point
    clipped = np.clip(bright, 0.0, 1.0)

    # 3. Bloom effect: create a glow map from the saturated areas.
    #    Saturated pixels (near 1.0) bleed light into neighbours via Gaussian blur.
    #    This mimics the optical blooming seen in real overexposed endoscopy frames.
    saturation_mask = np.clip(bright - 1.0, 0.0, None)   # excess light (>1.0)
    if saturation_mask.max() > 0.0:
        # Blur the saturation mask to create a soft halo
        glow_sigma = max(1.0, ev * 1.5)  # more blur for stronger overexposure
        glow = gaussian_filter(saturation_mask, sigma=glow_sigma)
        glow = glow / (glow.max() + 1e-7) * 0.25   # scale glow strength
        clipped = np.clip(clipped + glow, 0.0, 1.0)

    # 4. Slight colour desaturation near saturation (real sensor effect):
    #    pixels that clipped lose chromatic information → shift toward white
    luminance   = (0.299 * clipped[:, :, 0] + 0.587 * clipped[:, :, 1]
                   + 0.114 * clipped[:, :, 2])[:, :, None]
    desat_alpha = np.clip((bright.mean(axis=2, keepdims=True) - 0.8) / 0.4, 0.0, 0.15)
    clipped     = clipped * (1 - desat_alpha) + luminance * desat_alpha

    return clipped


def _simulate_underexposure(arr: np.ndarray, ev: float) -> np.ndarray:
    """
    Underexposure: linear attenuation + Poisson-Gaussian noise + vignetting.

    Poisson-Gaussian noise model:
        y = g * Poisson(x / g) + N(0, sigma_r^2)
    where:
        g     = camera gain (higher EV magnitude → lower effective gain)
        sigma_r = read noise (fixed floor, here ~0.005)

    For strongly underexposed images, the Poisson component dominates in
    shadows and the read noise creates a texture floor — distinct from the
    simple Gaussian model in the original code.
    """
    # 1. Linear attenuation
    attenuated = arr * (2.0 ** ev)

    # 2. Poisson-Gaussian noise
    #    Map ev → effective photon count (lower EV = fewer photons → more noise)
    gain     = max(0.002, 0.01 * (2.0 ** ev))   # effective photon scaling
    sigma_r  = 0.005                              # read noise floor

    # Poisson noise: draw from Poisson distribution scaled by gain
    photons     = attenuated / (gain + 1e-8)
    photons_obs = np.random.poisson(np.maximum(photons * 1000, 0)).astype(np.float32) / 1000
    poisson_noise = (photons_obs - photons) * gain

    # Gaussian read noise
    read_noise = np.random.normal(0.0, sigma_r, arr.shape).astype(np.float32)

    noisy = np.clip(attenuated + poisson_noise + read_noise, 0.0, 1.0)

    # 3. Vignetting: corners appear darker (natural in endoscope images)
    H, W = arr.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    # Normalised distance from centre [0, 1]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(x, y)
    dist   = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    # Vignette strength scales with underexposure magnitude
    vign_strength = min(0.35, abs(ev) * 0.08)
    vignette = (1.0 - vign_strength * dist ** 2)[..., None]  # (H, W, 1)
    noisy    = noisy * np.clip(vignette, 0.0, 1.0)

    return noisy

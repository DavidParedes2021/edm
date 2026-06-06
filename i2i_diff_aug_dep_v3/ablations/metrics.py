"""
ablations/metrics.py — Standalone metric implementations for the ablation
study of the depth-aware endoscopy diffusion pipeline.

Metric families:

  Reference-based (need a paired GT target_L on a held-out split):
      PSNR (L channel, on [0,100] range)
      SSIM (L channel)
      LPIPS (RGB, AlexNet backbone — falls back to L1-on-pixels if `lpips`
             is not installed; logged explicitly in the result JSON)

  Distribution-based (compare a folder of generated frames to an unpaired
  folder of REAL over/under endoscopy frames):
      FID  (Inception-V3 pool3 features, classic 2048-d)
      KID  (polynomial kernel, more stable for small samples — what you have)

  Hypothesis-targeted (small bespoke functions):
      delta_L_mask       — mean |L_out - L_src| inside the depth-expected
                           effect region. Effect *strength*.
      depth_mask_pearson — Pearson(ΔL, depth-expected mask). Effect
                           *placement* — key for the A1 (depth) ablation.
      delta_E00_AB       — mean CIEDE2000 between source AB and output AB.
                           Color preservation — key for A2 (L vs LAB).
      hf_pearson         — Pearson correlation of source vs output
                           high-frequency L. Texture preservation.
      sobel_l1           — L1 distance between source and output Sobel edge
                           magnitudes. Edge / texture preservation.

All metric functions accept numpy arrays in their natural ranges:
    rgb       : uint8 (H, W, 3)
    L         : float32 (H, W) in [0, 100]
    AB        : float32 (H, W, 2) in roughly [-128, 127]
    depth     : float32 (H, W) in [0, 1], 1 = near
    mask      : bool / float32 (H, W)

No internal Torch state is shared between calls except for the lazy
singletons created by `_lpips_model()` and `_inception_model()`, both of
which check CUDA availability once and cache the result.
"""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# Lazy globals
# ─────────────────────────────────────────────────────────────────────────────
_LPIPS = None
_INCEPTION = None
_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _DEVICE = "cpu"
    return _DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Reference-based pixel metrics
# ─────────────────────────────────────────────────────────────────────────────
def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 100.0) -> float:
    """PSNR in dB. pred/target on [0, data_range]."""
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    mse = float(np.mean((p - t) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 100.0) -> float:
    """SSIM on a 2-D float image; uses scikit-image if available, falls back
    to a NumPy implementation otherwise."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(pred, target, data_range=data_range))
    except ImportError:
        return _ssim_numpy(pred, target, data_range)


def ssim_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray,
             data_range: float = 255.0) -> float:
    """SSIM on an RGB image (multichannel, channel_axis=-1).

    Mirrors compute_paired_metrics.py exactly: scikit-image structural_similarity
    with channel_axis=-1 and data_range=255. Used for the source-referenced
    'how well is texture/structure preserved through the exposure edit?' metric.
    Falls back to a per-channel NumPy SSIM if scikit-image is unavailable.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(pred_rgb, target_rgb, channel_axis=-1,
                             data_range=data_range))
    except ImportError:
        vals = [_ssim_numpy(pred_rgb[..., c], target_rgb[..., c], data_range)
                for c in range(pred_rgb.shape[-1])]
        return float(np.mean(vals))


def _ssim_numpy(pred, target, data_range=100.0, win=11, k1=0.01, k2=0.03):
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    sigma = 1.5  # Gaussian standard for SSIM
    mu_p = gaussian_filter(p, sigma)
    mu_t = gaussian_filter(t, sigma)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t
    sig_p2 = gaussian_filter(p * p, sigma) - mu_p2
    sig_t2 = gaussian_filter(t * t, sigma) - mu_t2
    sig_pt = gaussian_filter(p * t, sigma) - mu_pt
    num = (2.0 * mu_pt + c1) * (2.0 * sig_pt + c2)
    den = (mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2)
    return float(np.mean(num / np.clip(den, 1e-12, None)))


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────
def _lpips_model():
    global _LPIPS
    if _LPIPS is not None:
        return _LPIPS
    try:
        import lpips  # noqa: F401
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _LPIPS = lpips.LPIPS(net="alex").to(_device()).eval()
    except ImportError:
        _LPIPS = False  # sentinel: explicitly unavailable
    return _LPIPS


def lpips_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    """LPIPS perceptual distance. Inputs: uint8 (H, W, 3).
    Falls back to per-pixel L1 in [0,1] if the `lpips` package isn't installed."""
    m = _lpips_model()
    if m is False:
        a = pred_rgb.astype(np.float32) / 255.0
        b = target_rgb.astype(np.float32) / 255.0
        return float(np.mean(np.abs(a - b)))
    import torch
    def _to_tensor(rgb):
        x = (rgb.astype(np.float32) / 127.5) - 1.0   # LPIPS expects [-1,1]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(_device())
        return x
    with torch.no_grad():
        v = m(_to_tensor(pred_rgb), _to_tensor(target_rgb)).item()
    return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis-targeted metrics
# ─────────────────────────────────────────────────────────────────────────────
def delta_L_mask(L_src: np.ndarray, L_out: np.ndarray,
                 mask: np.ndarray) -> float:
    """Mean |L_out - L_src| where mask > 0. Higher = stronger effect."""
    m = mask.astype(np.float32)
    w = m.sum()
    if w < 1.0:
        return 0.0
    diff = np.abs(L_out.astype(np.float32) - L_src.astype(np.float32))
    return float(np.sum(diff * m) / w)


def depth_mask_pearson(L_src: np.ndarray, L_out: np.ndarray,
                       depth: np.ndarray, mode: str,
                       gamma: float = 2.0) -> float:
    """Pearson correlation between (|L_out - L_src|) and the depth-expected
    effect mask. For 'underexposed' the expected mask peaks where (1-depth)
    is large; for 'overexposed' it peaks where depth is large."""
    if mode == "underexposed":
        expected = np.power(np.clip(1.0 - depth, 0.0, 1.0), gamma)
    else:
        expected = np.power(np.clip(depth, 0.0, 1.0), gamma)
    diff = np.abs(L_out.astype(np.float32) - L_src.astype(np.float32)).ravel()
    exp = expected.ravel().astype(np.float32)
    if diff.std() < 1e-6 or exp.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(diff, exp)[0, 1])


def delta_E00_AB(A_src: np.ndarray, B_src: np.ndarray,
                 A_out: np.ndarray, B_out: np.ndarray,
                 L_for_ref: np.ndarray) -> float:
    """Mean CIEDE2000 between (L, A_src, B_src) and (L, A_out, B_out).

    We hold L fixed at L_for_ref (typically L_src) so only the AB shift
    contributes — this is precisely what the L-vs-LAB ablation needs to
    measure (color drift, isolated from luminance changes). Returns the
    spatial mean ΔE_00.
    """
    L = L_for_ref.astype(np.float64)
    return float(np.mean(_ciede2000(L, A_src.astype(np.float64),
                                    B_src.astype(np.float64),
                                    L,
                                    A_out.astype(np.float64),
                                    B_out.astype(np.float64))))


def _ciede2000(L1, a1, b1, L2, a2, b2):
    """Vectorised CIEDE2000. Inputs are arrays of identical shape."""
    # Implementation follows Sharma, Wu & Dalal 2005 — using kL=kC=kH=1.
    L_bar = (L1 + L2) / 2.0
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - np.sqrt((C_bar ** 7) / (C_bar ** 7 + 25.0 ** 7 + 1e-12)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    Cp_bar = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    # When either chroma is zero, hue delta is undefined → 0.
    zero = (C1p * C2p) == 0
    dhp = np.where(zero, 0.0, dhp)
    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lp_bar = (L1 + L2) / 2.0
    Hp_sum = h1p + h2p
    Hp_bar = np.where(zero, Hp_sum,
                      np.where(np.abs(h1p - h2p) > 180.0,
                               (Hp_sum + 360.0) / 2.0,
                               Hp_sum / 2.0))
    T = (1.0
         - 0.17 * np.cos(np.radians(Hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * Hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * Hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * Hp_bar - 63.0)))
    dtheta = 30.0 * np.exp(-(((Hp_bar - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt((Cp_bar ** 7) / (Cp_bar ** 7 + 25.0 ** 7 + 1e-12))
    Sl = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc
    dE2 = ((dLp / Sl) ** 2
           + (dCp / Sc) ** 2
           + (dHp / Sh) ** 2
           + Rt * (dCp / Sc) * (dHp / Sh))
    return np.sqrt(np.clip(dE2, 0.0, None))


def hf_pearson(L_src: np.ndarray, L_out: np.ndarray, sigma: float = 3.0) -> float:
    """Pearson correlation between the high-frequency components of source
    and output L. Detects whether the model preserved fine texture."""
    src_hf = L_src - gaussian_filter(L_src.astype(np.float32), sigma=sigma)
    out_hf = L_out - gaussian_filter(L_out.astype(np.float32), sigma=sigma)
    a = src_hf.ravel().astype(np.float64)
    b = out_hf.ravel().astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sobel_l1(L_src: np.ndarray, L_out: np.ndarray) -> float:
    """L1 distance between Sobel edge magnitudes of source and output L."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from scipy.signal import convolve2d
    def _edges(x):
        gx = convolve2d(x.astype(np.float32), kx, mode="same", boundary="symm")
        gy = convolve2d(x.astype(np.float32), ky, mode="same", boundary="symm")
        return np.sqrt(gx * gx + gy * gy)
    return float(np.mean(np.abs(_edges(L_src) - _edges(L_out))))


# ─────────────────────────────────────────────────────────────────────────────
# FID / KID (Inception-V3 pool3 features)
# ─────────────────────────────────────────────────────────────────────────────
def _inception_model():
    """Load Inception-V3 once. Removes the final classifier and exposes the
    2048-d pool3 features used by the standard FID/KID formulations."""
    global _INCEPTION
    if _INCEPTION is not None:
        return _INCEPTION
    import torch
    from torchvision.models import inception_v3
    try:
        # torchvision >= 0.13: enum-based weights API
        from torchvision.models import Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           aux_logits=True)
    except ImportError:
        # torchvision < 0.13: legacy pretrained flag
        net = inception_v3(pretrained=True, aux_logits=True)
    net.fc = torch.nn.Identity()
    net.eval()
    _INCEPTION = net.to(_device())
    return _INCEPTION


def _inception_features(images: Iterable[np.ndarray],
                        batch_size: int = 16) -> np.ndarray:
    """Compute Inception pool3 features for a list/iterable of HxWx3 uint8
    images. Returns an (N, 2048) numpy array."""
    import torch
    import torch.nn.functional as F
    net = _inception_model()
    dev = _device()
    feats = []
    buf = []

    def _flush():
        if not buf:
            return
        # Images may have different resolutions, so resize each to 299x299
        # individually before batching (Inception's input size anyway).
        chw = []
        for img in buf:
            t = torch.from_numpy(img.astype(np.float32) / 255.0)  # HWC [0,1]
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)           # 1CHW
            t = F.interpolate(t, size=(299, 299), mode="bilinear",
                              align_corners=False, antialias=True)
            chw.append(t)
        x = torch.cat(chw, dim=0)                                  # NCHW
        # Inception normalisation: ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            f = net(x)
        feats.append(f.cpu().numpy())
        buf.clear()

    for img in images:
        buf.append(img)
        if len(buf) >= batch_size:
            _flush()
    _flush()
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048))


def fid_score(feats_p: np.ndarray, feats_t: np.ndarray) -> float:
    """Fréchet distance between two 2048-d Gaussian-summarised feature sets."""
    from scipy.linalg import sqrtm
    if feats_p.shape[0] < 2 or feats_t.shape[0] < 2:
        return float("nan")
    mu_p, mu_t = feats_p.mean(axis=0), feats_t.mean(axis=0)
    sig_p = np.cov(feats_p, rowvar=False)
    sig_t = np.cov(feats_t, rowvar=False)
    covmean, _ = sqrtm(sig_p @ sig_t, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_p - mu_t
    return float(diff @ diff + np.trace(sig_p + sig_t - 2.0 * covmean))


def kid_score(feats_p: np.ndarray, feats_t: np.ndarray,
              num_subsets: int = 100, subset_size: int = 100,
              seed: int = 0) -> float:
    """Polynomial-kernel KID (Bińkowski et al. 2018). Stable for small
    sample sizes (< 1000). Returns the mean across `num_subsets` resamples."""
    if feats_p.shape[0] < 10 or feats_t.shape[0] < 10:
        return float("nan")
    n = min(subset_size, feats_p.shape[0], feats_t.shape[0])
    d = feats_p.shape[1]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_subsets):
        i = rng.choice(feats_p.shape[0], n, replace=False)
        j = rng.choice(feats_t.shape[0], n, replace=False)
        x, y = feats_p[i], feats_t[j]
        # Polynomial kernel from the paper: (x·y / d + 1)^3
        kxx = ((x @ x.T) / d + 1.0) ** 3
        kyy = ((y @ y.T) / d + 1.0) ** 3
        kxy = ((x @ y.T) / d + 1.0) ** 3
        # Unbiased MMD^2 estimator
        m = n
        kxx_sum = kxx.sum() - np.trace(kxx)
        kyy_sum = kyy.sum() - np.trace(kyy)
        mmd2 = (kxx_sum / (m * (m - 1))
                + kyy_sum / (m * (m - 1))
                - 2.0 * kxy.mean())
        out.append(mmd2)
    return float(np.mean(out))


def fid_from_folders(gen_dir: str, ref_dir: str,
                     batch_size: int = 16,
                     max_images: int | None = None) -> dict:
    """Compute FID and KID between two folders of images.

    Returns: {"fid": float, "kid": float, "n_gen": int, "n_ref": int}."""
    def _load_dir(d):
        files = []
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            files.extend(sorted(Path(d).glob(f"*{ext}")))
            files.extend(sorted(Path(d).glob(f"*{ext.upper()}")))
        if max_images:
            files = files[:max_images]
        imgs = []
        for f in files:
            img = np.array(Image.open(f).convert("RGB"))
            imgs.append(img)
        return imgs

    gen_imgs = _load_dir(gen_dir)
    ref_imgs = _load_dir(ref_dir)
    f_gen = _inception_features(gen_imgs, batch_size=batch_size)
    f_ref = _inception_features(ref_imgs, batch_size=batch_size)
    return {
        "fid":   fid_score(f_gen, f_ref),
        "kid":   kid_score(f_gen, f_ref),
        "n_gen": int(f_gen.shape[0]),
        "n_ref": int(f_ref.shape[0]),
    }

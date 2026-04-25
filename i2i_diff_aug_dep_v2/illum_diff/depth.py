"""Endoscopy-specific photometric depth proxy + focal-cluster mask generation.

Why a depth *proxy* and not MiDaS:
    The light source travels with the endoscope camera, so under inverse-square
    illumination falloff: brightness ~ 1 / depth^2. A blurred -log(L) of the
    image therefore behaves as a smooth inverse-depth proxy. This avoids
    pulling in a heavy off-the-shelf depth network and matches the very cue
    that drives natural exposure artifacts in the data.

Masks (always a single, largest connected component, then dilated and softened):
    - Overexposure focal cluster -> the *closest* region (lowest depth_proxy /
      brightest pixels), which is where saturation actually appears.
    - Underexposure focal cluster -> the *farthest* region (highest depth_proxy
      / darkest pixels), which is the lumen / shadowed area.

The same mask routines are reused at training (where the score is taken from
the artifact image's L channel) and at inference (where it is taken from the
normal frame's depth proxy), keeping training/inference distributions aligned.
"""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label
from skimage.color import rgb2gray


def valid_frame_mask(rgb_uint8: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Boolean mask of pixels that are NOT in the black circular vignette."""
    s = rgb_uint8.astype(np.int32).sum(axis=-1)
    return s > threshold


def depth_proxy(rgb_uint8: np.ndarray, blur_sigma: float = 8.0,
                eps: float = 1e-3) -> np.ndarray:
    """Photometric inverse-illumination depth cue, normalized to [0, 1].

    Higher value <=> farther from the endoscope tip.
    """
    gray = rgb2gray(rgb_uint8).astype(np.float32)         # in [0, 1]
    smoothed = gaussian_filter(gray, sigma=blur_sigma)
    smoothed = np.clip(smoothed, eps, 1.0)
    inv_log = -np.log(smoothed).astype(np.float32)        # darker -> larger
    valid = valid_frame_mask(rgb_uint8)
    if valid.sum() == 0:
        return np.zeros_like(inv_log, dtype=np.float32)
    vmin = float(inv_log[valid].min())
    vmax = float(inv_log[valid].max())
    proxy = (inv_log - vmin) / max(vmax - vmin, 1e-8)
    proxy = np.clip(proxy, 0.0, 1.0).astype(np.float32)
    proxy = proxy * valid.astype(np.float32)
    return proxy


def _largest_connected_component(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.sum() == 0:
        return mask_bool
    labels, n = cc_label(mask_bool)
    if n <= 1:
        return mask_bool
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def make_focal_mask(score_map: np.ndarray, valid: np.ndarray,
                    fraction: float, dilation_iters: int,
                    blur_sigma: float, min_area_frac: float) -> np.ndarray:
    """Build a soft (0..1) focal-cluster mask:

        threshold the top `fraction` of valid pixels of `score_map`
        -> keep largest connected component
        -> binary-dilate
        -> Gaussian-soften the boundary
        -> mask away the vignette

    Higher `score_map` values are interpreted as "more focal".
    """
    out = np.zeros_like(score_map, dtype=np.float32)
    valid_scores = score_map[valid]
    if valid_scores.size == 0:
        return out
    k = int(np.ceil(fraction * valid_scores.size))
    if k <= 0:
        return out
    # k-th largest threshold
    thresh = np.partition(valid_scores, -k)[-k]
    binary = (score_map >= thresh) & valid
    binary = _largest_connected_component(binary)
    if binary.sum() < min_area_frac * max(int(valid.sum()), 1):
        return out
    if dilation_iters > 0:
        binary = binary_dilation(binary, iterations=int(dilation_iters))
    soft = gaussian_filter(binary.astype(np.float32), sigma=float(blur_sigma))
    m = float(soft.max())
    if m > 0:
        soft = soft / m
    soft = soft * valid.astype(np.float32)
    return soft.astype(np.float32)


# ----- Mask drivers: artifact-frame (training) -----------------------------------

def overexposure_mask_from_artifact(L_0_100: np.ndarray, valid: np.ndarray,
                                    fraction: float, dilation_iters: int,
                                    blur_sigma: float, min_area_frac: float) -> np.ndarray:
    """For training on overexposed frames -- the focal cluster IS the bright cluster."""
    score = L_0_100.astype(np.float32) * valid.astype(np.float32)
    return make_focal_mask(score, valid, fraction, dilation_iters, blur_sigma, min_area_frac)


def underexposure_mask_from_artifact(L_0_100: np.ndarray, valid: np.ndarray,
                                     fraction: float, dilation_iters: int,
                                     blur_sigma: float, min_area_frac: float) -> np.ndarray:
    """For training on underexposed frames -- the focal cluster IS the dark cluster."""
    score = (100.0 - L_0_100.astype(np.float32)) * valid.astype(np.float32)
    return make_focal_mask(score, valid, fraction, dilation_iters, blur_sigma, min_area_frac)


# ----- Mask drivers: depth-proxy (inference on normal frames) --------------------

def overexposure_mask_from_normal(rgb_uint8: np.ndarray, fraction: float,
                                  dilation_iters: int, blur_sigma: float,
                                  min_area_frac: float) -> np.ndarray:
    """Closest cluster (low depth) -- where saturation will be painted in."""
    valid = valid_frame_mask(rgb_uint8)
    d = depth_proxy(rgb_uint8)
    score = (1.0 - d) * valid.astype(np.float32)
    return make_focal_mask(score, valid, fraction, dilation_iters, blur_sigma, min_area_frac)


def underexposure_mask_from_normal(rgb_uint8: np.ndarray, fraction: float,
                                   dilation_iters: int, blur_sigma: float,
                                   min_area_frac: float) -> np.ndarray:
    """Farthest cluster (high depth) -- where shadow/lumen will be painted in."""
    valid = valid_frame_mask(rgb_uint8)
    d = depth_proxy(rgb_uint8)
    score = d * valid.astype(np.float32)
    return make_focal_mask(score, valid, fraction, dilation_iters, blur_sigma, min_area_frac)

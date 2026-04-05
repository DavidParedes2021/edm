"""
evaluation/metrics.py
Evaluation metrics for generated illumination artifacts.

Metrics:
  PSNR   – peak signal-to-noise ratio          (higher = better)
  SSIM   – structural similarity               (higher = better, max 1)
  LPIPS  – learned perceptual image patch sim  (lower = better)
  EBS    – exposure bias score                 (luminance difference from GT)
  FID    – Fréchet inception distance          (lower = better, distribution quality)

Usage:
  python evaluation/metrics.py \\
      --generated  path/to/generated/overexposed/ \\
      --reference  path/to/real_overexposed/test/overexposed/ \\
      --normal     path/to/real_overexposed/test/normal_frames/ \\
      --output     evaluation/results.json
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_image_np(path: str, size: Optional[int] = None) -> np.ndarray:
    """Load image as float32 numpy array in [0, 1], shape [H, W, 3]."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0


def load_image_tensor(path: str, size: Optional[int] = None) -> torch.Tensor:
    """Load image as tensor [1, 3, H, W] in [0, 1]."""
    img = load_image_np(path, size)
    t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def collect_pairs(dir_a: str, dir_b: str) -> List[Tuple[str, str]]:
    """Match files by name between two directories."""
    files_a = {Path(f).stem: str(f) for f in Path(dir_a).iterdir()
               if Path(f).suffix.lower() in VALID_EXT}
    files_b = {Path(f).stem: str(f) for f in Path(dir_b).iterdir()
               if Path(f).suffix.lower() in VALID_EXT}
    common = sorted(set(files_a) & set(files_b))
    return [(files_a[k], files_b[k]) for k in common]


# ──────────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR between two [H,W,3] float32 arrays in [0,1]."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10 * math.log10(1.0 / mse))


# ──────────────────────────────────────────────────────────────────────────────
# SSIM  (pure numpy, no skimage dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _ssim_channel(x: np.ndarray, y: np.ndarray, K1=0.01, K2=0.03, win=11) -> float:
    """SSIM for a single-channel 2D array in [0,1]."""
    from scipy.ndimage import uniform_filter
    C1 = K1 ** 2
    C2 = K2 ** 2

    mu_x  = uniform_filter(x, win)
    mu_y  = uniform_filter(y, win)
    mu_xx = uniform_filter(x * x, win)
    mu_yy = uniform_filter(y * y, win)
    mu_xy = uniform_filter(x * y, win)

    sigma_x  = mu_xx - mu_x ** 2
    sigma_y  = mu_yy - mu_y ** 2
    sigma_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float(np.mean(num / (den + 1e-10)))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean SSIM over RGB channels, arrays in [H,W,3]."""
    try:
        scores = [_ssim_channel(pred[..., c], gt[..., c]) for c in range(3)]
        return float(np.mean(scores))
    except ImportError:
        # scipy unavailable – fall back to a simple correlation-based approximation
        pred_f = pred.flatten()
        gt_f   = gt.flatten()
        cov    = np.cov(pred_f, gt_f)[0, 1]
        return float(np.clip(cov / (pred_f.std() * gt_f.std() + 1e-8), -1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS  (VGG-based, simplified)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleLPIPS:
    """
    Lightweight LPIPS approximation using VGG16 relu features.
    Does NOT require the lpips package.
    """

    def __init__(self, device: torch.device):
        self.device = device
        try:
            import torchvision.models as tvm
            vgg = tvm.vgg16(pretrained=True).features.to(device).eval()
            self.slices = [
                vgg[:4],   # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16], # relu3_3
                vgg[16:23],# relu4_3
            ]
            for s in self.slices:
                for p in s.parameters():
                    p.requires_grad_(False)
            self._available = True
        except Exception:
            self._available = False

        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @torch.no_grad()
    def __call__(self, pred_t: torch.Tensor, gt_t: torch.Tensor) -> float:
        """pred_t, gt_t: [1,3,H,W] in [0,1]"""
        if not self._available:
            return float("nan")

        pred_t = self.normalize(pred_t.to(self.device))
        gt_t   = self.normalize(gt_t.to(self.device))

        total = 0.0
        p, g  = pred_t, gt_t
        for sl in self.slices:
            p = sl(p)
            g = sl(g)
            # Normalise feature maps
            pn = F.normalize(p, dim=1)
            gn = F.normalize(g, dim=1)
            total += F.mse_loss(pn, gn).item()

        return total / len(self.slices)


# ──────────────────────────────────────────────────────────────────────────────
# Exposure Bias Score (EBS)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ebs(generated: np.ndarray, normal: np.ndarray, gt_artifact: np.ndarray) -> dict:
    """
    Measures whether generated image has the right *direction* and *amount*
    of exposure shift relative to normal.

    Returns:
      ebs_direction: cosine between (gen-normal) and (gt-normal) luminance vecs.
                     1.0 = perfect, -1.0 = inverted.
      ebs_magnitude: ratio of exposure magnitudes. 1.0 = same intensity as GT.
      lum_gen:       mean luminance of generated artifact.
      lum_gt:        mean luminance of real GT artifact.
      lum_normal:    mean luminance of normal frame.
    """
    def lum(img):
        # Perceived luminance (BT.601)
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    l_gen    = lum(generated).mean()
    l_gt     = lum(gt_artifact).mean()
    l_normal = lum(normal).mean()

    shift_gen = l_gen - l_normal
    shift_gt  = l_gt  - l_normal

    direction = (shift_gen * shift_gt) / (abs(shift_gen) * abs(shift_gt) + 1e-8)
    magnitude = abs(shift_gen) / (abs(shift_gt) + 1e-8)

    return {
        "ebs_direction":  float(direction),
        "ebs_magnitude":  float(magnitude),
        "lum_generated":  float(l_gen),
        "lum_gt":         float(l_gt),
        "lum_normal":     float(l_normal),
        "lum_shift_gen":  float(shift_gen),
        "lum_shift_gt":   float(shift_gt),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FID  (using Inception V3 features, simplified)
# ──────────────────────────────────────────────────────────────────────────────

def _get_inception_features(
    image_paths: list,
    device: torch.device,
    resize: int = 299,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract Inception-v3 pool features for FID."""
    import torchvision.models as tvm

    inception = tvm.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # remove final classifier
    inception.eval()

    features = []
    transform = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception features"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            feat = inception(batch)
            if isinstance(feat, tuple):
                feat = feat[0]
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(
    gen_dir: str,
    ref_dir: str,
    device: torch.device,
    max_images: int = 500,
) -> float:
    """
    Compute FID between generated and reference directories.
    Uses simplified closed-form FID (Heusel et al.).
    """
    def list_images(d):
        return sorted([str(p) for p in Path(d).iterdir()
                       if Path(p).suffix.lower() in VALID_EXT])[:max_images]

    gen_paths = list_images(gen_dir)
    ref_paths = list_images(ref_dir)

    if len(gen_paths) < 10 or len(ref_paths) < 10:
        print(f"[WARN] Not enough images for FID (gen={len(gen_paths)}, ref={len(ref_paths)}). Skipping.")
        return float("nan")

    try:
        feat_gen = _get_inception_features(gen_paths, device)
        feat_ref = _get_inception_features(ref_paths, device)
    except Exception as e:
        print(f"[WARN] FID computation failed: {e}")
        return float("nan")

    mu1, sig1 = feat_gen.mean(0), np.cov(feat_gen, rowvar=False)
    mu2, sig2 = feat_ref.mean(0), np.cov(feat_ref, rowvar=False)

    diff = mu1 - mu2
    # Matrix square root via eigendecomposition (numerically stable)
    try:
        vals, vecs = np.linalg.eigh(sig1 @ sig2)
        vals = np.maximum(vals, 0)
        sqrt_sig1sig2 = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        fid = float(diff @ diff + np.trace(sig1 + sig2 - 2 * sqrt_sig1sig2))
    except np.linalg.LinAlgError:
        fid = float("nan")

    return fid


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    generated_dir: str,
    reference_dir: str,
    normal_dir:    str,
    output_path:   str,
    image_size:    int  = 256,
    compute_fid_:  bool = True,
    device_str:    str  = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    lpips  = SimpleLPIPS(device)

    gen_ref_pairs  = collect_pairs(generated_dir, reference_dir)
    gen_norm_pairs = collect_pairs(generated_dir, normal_dir)

    if len(gen_ref_pairs) == 0:
        print("[ERROR] No matching files between generated and reference directories.")
        return

    # Build normal lookup
    norm_lookup = {Path(a).stem: b for a, b in gen_norm_pairs}

    psnr_list, ssim_list, lpips_list, ebs_dir_list, ebs_mag_list = [], [], [], [], []

    for gen_path, ref_path in tqdm(gen_ref_pairs, desc="Per-image metrics"):
        stem = Path(gen_path).stem
        gen_np = load_image_np(gen_path, image_size)
        ref_np = load_image_np(ref_path, image_size)

        psnr_list.append(compute_psnr(gen_np, ref_np))
        ssim_list.append(compute_ssim(gen_np, ref_np))

        gen_t = load_image_tensor(gen_path, image_size)
        ref_t = load_image_tensor(ref_path, image_size)
        lpips_list.append(lpips(gen_t, ref_t))

        if stem in norm_lookup:
            norm_np = load_image_np(norm_lookup[stem], image_size)
            ebs     = compute_ebs(gen_np, norm_np, ref_np)
            ebs_dir_list.append(ebs["ebs_direction"])
            ebs_mag_list.append(ebs["ebs_magnitude"])

    results = {
        "n_images":      len(gen_ref_pairs),
        "PSNR":          {"mean": float(np.mean(psnr_list)),  "std": float(np.std(psnr_list))},
        "SSIM":          {"mean": float(np.mean(ssim_list)),  "std": float(np.std(ssim_list))},
        "LPIPS":         {"mean": float(np.nanmean(lpips_list)), "std": float(np.nanstd(lpips_list))},
        "EBS_direction": {"mean": float(np.mean(ebs_dir_list)) if ebs_dir_list else float("nan"),
                          "std":  float(np.std(ebs_dir_list))  if ebs_dir_list else float("nan")},
        "EBS_magnitude": {"mean": float(np.mean(ebs_mag_list)) if ebs_mag_list else float("nan"),
                          "std":  float(np.std(ebs_mag_list))  if ebs_mag_list else float("nan")},
    }

    if compute_fid_:
        print("Computing FID...")
        results["FID"] = compute_fid(generated_dir, reference_dir, device)

    # Print summary
    print("\n" + "=" * 50)
    print(" Evaluation Results")
    print("=" * 50)
    for metric, val in results.items():
        if isinstance(val, dict) and "mean" in val:
            print(f"  {metric:20s}: {val['mean']:.4f}  ± {val['std']:.4f}")
        else:
            print(f"  {metric:20s}: {val}")
    print("=" * 50)

    # Save JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate generated illumination artifacts")
    p.add_argument("--generated",  required=True, help="Dir with generated artifact images")
    p.add_argument("--reference",  required=True, help="Dir with real artifact images (GT)")
    p.add_argument("--normal",     required=True, help="Dir with corresponding normal frames")
    p.add_argument("--output",     default="evaluation/results.json")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--no_fid",     action="store_true", help="Skip slow FID computation")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        generated_dir = args.generated,
        reference_dir = args.reference,
        normal_dir    = args.normal,
        output_path   = args.output,
        image_size    = args.image_size,
        compute_fid_  = not args.no_fid,
        device_str    = args.device,
    )

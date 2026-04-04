"""
evaluate.py — Evaluation metrics for the synthetic illumination dataset.

Metrics computed:
  1. FID   — distributional realism vs. real target domain images
  2. LPIPS — geometry preservation vs. input Normal frames
  3. SSIM-Y — structural similarity on luminance channel
  4. ΔHistogram — mean pixel intensity shift vs. target domain
  5. Exposure accuracy — fraction of images with mean luminance in target range

Usage:
    python evaluate.py \
        --generated synthetic/overexposed \
        --normal    data/normal \
        --real      data/overexposed \
        --domain    over
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.models as tv_models
import torchvision.transforms as T

from dataset import list_images, pil_to_tensor
from model   import SSIMLoss, LPIPSLoss
from config  import TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True)
    p.add_argument("--normal",    required=True)
    p.add_argument("--real",      required=True,
                   help="Real domain B or C images for FID reference.")
    p.add_argument("--domain",    choices=["over", "under"], default="over")
    p.add_argument("--batch",     type=int, default=8)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# FID (simplified — InceptionV3 features)
# ──────────────────────────────────────────────────────────────────────────────

def _inception_features(paths, device, batch_size=8, size=299):
    """Extract InceptionV3 pool3 features from a list of image paths."""
    model = tv_models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            imgs  = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in chunk
            ]).to(device)
            f = model(imgs)
            feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)


def compute_fid(feats_gen, feats_real):
    """Fréchet Inception Distance between two feature arrays."""
    from scipy import linalg

    mu_g, sg = feats_gen.mean(0),  np.cov(feats_gen, rowvar=False)
    mu_r, sr = feats_real.mean(0), np.cov(feats_real, rowvar=False)

    diff = mu_g - mu_r
    covmean, _ = linalg.sqrtm(sg.dot(sr), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff.dot(diff) + np.trace(sg + sr - 2 * covmean))
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# Luminance helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_luminance(img_tensor: torch.Tensor) -> float:
    """(3,H,W) [-1,1] → mean luminance in [0,1]"""
    t01 = (img_tensor + 1.0) / 2.0
    Y   = (0.299 * t01[0] + 0.587 * t01[1] + 0.114 * t01[2])
    return float(Y.mean().item())


def histogram_distance(gen: torch.Tensor, ref: torch.Tensor, bins=64) -> float:
    """L1 distance between histograms of two (3,H,W) tensors."""
    def hist(t):
        t01 = ((t + 1.0) / 2.0).flatten().numpy()
        h, _ = np.histogram(t01, bins=bins, range=(0, 1), density=True)
        return h / (h.sum() + 1e-7)
    return float(np.abs(hist(gen) - hist(ref)).mean())


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate():
    args   = parse_args()
    cfg    = TrainConfig()
    device = cfg.device

    gen_paths    = list_images(args.generated)
    normal_paths = list_images(args.normal)
    real_paths   = list_images(args.real)

    # Match generated ↔ normal by index (assumes same ordering/filenames)
    n_eval = min(len(gen_paths), len(normal_paths))
    gen_paths    = gen_paths[:n_eval]
    normal_paths = normal_paths[:n_eval]

    print(f"Evaluating {n_eval} generated frames against {len(real_paths)} real samples.")

    lpips_fn = LPIPSLoss(device=str(device))
    ssim_fn  = SSIMLoss(device=str(device))

    lpips_scores, ssim_scores, hist_dists, lum_values = [], [], [], []

    for i, (gp, np_) in enumerate(zip(gen_paths, normal_paths)):
        gen_t    = pil_to_tensor(Image.open(gp).convert("RGB"),  cfg.image_size)
        norm_t   = pil_to_tensor(Image.open(np_).convert("RGB"), cfg.image_size)
        gen_b    = gen_t.unsqueeze(0).to(device)
        norm_b   = norm_t.unsqueeze(0).to(device)

        # Random real reference for histogram distance
        ref_t    = pil_to_tensor(
            Image.open(real_paths[i % len(real_paths)]).convert("RGB"),
            cfg.image_size
        )

        with torch.no_grad():
            lp = lpips_fn(gen_b, norm_b).item()
            ss = ssim_fn(gen_b, norm_b).item()    # 1 - SSIM, lower = better

        hd  = histogram_distance(gen_t.cpu(), ref_t.cpu())
        lum = mean_luminance(gen_t.cpu())

        lpips_scores.append(lp)
        ssim_scores.append(ss)
        hist_dists.append(hd)
        lum_values.append(lum)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_eval}]", end="\r")

    # ── FID ──────────────────────────────────────────────────────────────────
    print("\nComputing FID (InceptionV3 features)...")
    feats_gen  = _inception_features(gen_paths,  device, batch_size=args.batch)
    feats_real = _inception_features(real_paths, device, batch_size=args.batch)
    fid        = compute_fid(feats_gen, feats_real)

    # ── Exposure accuracy ─────────────────────────────────────────────────────
    if args.domain == "over":
        # Overexposed: mean luminance should be > 0.55
        in_range = sum(1 for l in lum_values if l > 0.55) / n_eval
    else:
        # Underexposed: mean luminance should be < 0.45
        in_range = sum(1 for l in lum_values if l < 0.45) / n_eval

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  FID               : {fid:.2f}       (lower = more realistic)")
    print(f"  LPIPS (geom.)     : {np.mean(lpips_scores):.4f}  (lower = better geometry)")
    print(f"  1-SSIM (luma)     : {np.mean(ssim_scores):.4f}  (lower = better structure)")
    print(f"  ΔHistogram        : {np.mean(hist_dists):.4f}  (lower = better EV match)")
    print(f"  Exposure accuracy : {in_range * 100:.1f}%  (frames in target lum range)")
    print(f"  Mean luminance    : {np.mean(lum_values):.3f}")
    print("=" * 60)

    return {
        "fid":    fid,
        "lpips":  float(np.mean(lpips_scores)),
        "ssim":   float(np.mean(ssim_scores)),
        "hist":   float(np.mean(hist_dists)),
        "exp_acc": in_range,
        "mean_lum": float(np.mean(lum_values)),
    }


if __name__ == "__main__":
    evaluate()

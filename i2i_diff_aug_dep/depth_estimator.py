#!/usr/bin/env python3
"""
depth_estimator.py — Precompute monocular depth maps for the normal-frames set.

Backends
--------
  midas                 Default. torch.hub MiDaS DPT_Large.
                        Works with any transformers version; only needs torch.
                        Inverse depth, higher = closer (same convention as DA V2).
  midas_hybrid          Faster, slightly lower quality MiDaS DPT_Hybrid.
  depth_anything_v2_hf  HuggingFace transformers path. Needs transformers >= 4.42.
                        Will error out politely on older versions.

Output
------
  One float16 .npy per input image, per-image min-max normalised to [0, 1]:
      1.0 = nearest surface
      0.0 = farthest surface
  Plus a greyscale PNG preview in <o>/previews/ for quick sanity checks.

Usage
-----
  # Default MiDaS — works right now regardless of transformers version
  python depth_estimator.py \\
      --input_dir  ../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames \\
      --output_dir ../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/depth

  # Depth Anything V2 — requires upgrading transformers first
  python depth_estimator.py --backend depth_anything_v2_hf --input_dir ... --output_dir ...

  # Faster MiDaS variant for quick experiments
  python depth_estimator.py --backend midas_hybrid --input_dir ... --output_dir ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ═══════════════════════════════════════════════════════════════════════════
# Backend loaders
# ═══════════════════════════════════════════════════════════════════════════

def _load_midas(model_type: str, device: torch.device):
    """Load a MiDaS model via torch.hub. Returns (model, transform_fn)."""
    print(f"[depth] loading MiDaS ({model_type}) via torch.hub …")
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model = model.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return model, transform


def _load_depth_anything_v2_hf(device: torch.device):
    """Load Depth Anything V2 Small via HF transformers pipeline.
    Raises a clear error if the installed transformers is too old."""
    try:
        import transformers
    except ImportError:
        sys.exit("[depth] transformers is not installed.")

    # DA V2 was added to transformers on 2024-07-05 → need >= 4.42
    try:
        parts = [int(p) for p in transformers.__version__.split(".")[:2]]
        major, minor = parts[0], parts[1]
    except Exception:
        major, minor = 0, 0
    if (major, minor) < (4, 42):
        sys.exit(
            f"[depth] transformers {transformers.__version__} is too old for "
            f"Depth Anything V2 (need >= 4.42).\n"
            f"        Either upgrade (`pip install -U 'transformers>=4.44'`) or "
            f"use --backend midas."
        )

    from transformers import pipeline
    print("[depth] loading Depth Anything V2 Small via HF pipeline …")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device.type == "cuda" else -1,
    )
    return pipe, None


# ═══════════════════════════════════════════════════════════════════════════
# Per-image inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _infer_midas(model, transform, img_pil: Image.Image,
                 device: torch.device) -> np.ndarray:
    """Run MiDaS on a PIL image → (H, W) float32, min-max normalised."""
    img_np = np.array(img_pil.convert("RGB"))
    H, W = img_np.shape[:2]
    # MiDaS transforms expect a numpy uint8 RGB array, return a tensor
    batch = transform(img_np).to(device)
    pred = model(batch)
    # resize to original resolution
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.detach().cpu().numpy().astype(np.float32)
    return _normalize_01(d)


@torch.no_grad()
def _infer_da_v2_hf(pipe, _unused, img_pil: Image.Image,
                    device: torch.device) -> np.ndarray:
    """Run HF DA V2 pipeline → (H, W) float32, min-max normalised."""
    out = pipe(img_pil)
    # HF depth-estimation pipeline returns {'depth': PIL image, 'predicted_depth': tensor}
    # 'predicted_depth' is inverse depth in the model's native resolution.
    pred = out["predicted_depth"].to(device)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    W, H = img_pil.size  # PIL gives (W, H)
    pred = torch.nn.functional.interpolate(
        pred, size=(H, W), mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.detach().cpu().numpy().astype(np.float32)
    return _normalize_01(d)


def _normalize_01(d: np.ndarray) -> np.ndarray:
    """Per-image min-max normalise to [0, 1]. 1.0 = nearest, 0.0 = farthest."""
    mn, mx = float(d.min()), float(d.max())
    if mx - mn < 1e-6:
        return np.full_like(d, 0.5, dtype=np.float32)
    return ((d - mn) / (mx - mn)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", default="midas",
                        choices=["midas", "midas_hybrid", "depth_anything_v2_hf"])
    parser.add_argument("--save_previews", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run even if .npy already exists")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = out_dir / "previews"
    if args.save_previews:
        prev_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth] device = {device}")

    # ── load backend ────────────────────────────────────────────────────
    if args.backend == "midas":
        model, transform = _load_midas("DPT_Large", device)
        infer = _infer_midas
    elif args.backend == "midas_hybrid":
        model, transform = _load_midas("DPT_Hybrid", device)
        infer = _infer_midas
    else:  # depth_anything_v2_hf
        model, transform = _load_depth_anything_v2_hf(device)
        infer = _infer_da_v2_hf

    # ── iterate ─────────────────────────────────────────────────────────
    paths = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"[depth] no images found in {in_dir}")

    print(f"[depth] processing {len(paths)} images …")
    for p in tqdm(paths):
        out_npy = out_dir / f"{p.stem}.npy"
        if out_npy.exists() and not args.overwrite:
            continue

        img = Image.open(p).convert("RGB")
        d = infer(model, transform, img, device)  # (H, W) float32 in [0, 1]
        np.save(out_npy, d.astype(np.float16))

        if args.save_previews:
            prev = (d * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(prev, mode="L").save(prev_dir / f"{p.stem}.png")

    print(f"[depth] done — {len(paths)} maps written to {out_dir}")


if __name__ == "__main__":
    main()
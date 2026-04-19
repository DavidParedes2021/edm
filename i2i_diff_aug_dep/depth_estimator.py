#!/usr/bin/env python3
"""
depth_estimator.py — Precompute monocular depth maps for all normal frames.

Uses Depth Anything V2 (Small) via Hugging Face Transformers. Depth is stored
as float16 .npy at the ORIGINAL image resolution — the dataset loaders resize
it on-the-fly. Depth is normalized per-image to [0, 1] where:
    1.0 = nearest (camera side, overexposure candidate)
    0.0 = farthest (cavity, underexposure candidate)

Note on Depth Anything's output convention:
    The model returns RELATIVE INVERSE DEPTH — larger raw values = closer.
    After per-image min-max to [0, 1], 1.0 is correctly the nearest point.
    We keep this convention throughout the pipeline.

Usage:
    python depth_estimator.py \\
        --input_dir ./data/normal \\
        --output_dir ./data/depth

    # Use base variant if Small produces noisy maps on your data:
    python depth_estimator.py \\
        --input_dir ./data/normal \\
        --output_dir ./data/depth \\
        --model_id depth-anything/Depth-Anything-V2-Base-hf
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_depth_pipeline(model_id: str, device: torch.device):
    """Build a Depth Anything V2 pipeline via transformers."""
    from transformers import pipeline
    print(f"[Depth] loading {model_id} on {device}")
    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device=0 if device.type == "cuda" else -1,
    )
    return pipe


@torch.no_grad()
def estimate_depth(pipe, img_pil: Image.Image) -> np.ndarray:
    """Run depth estimation on one PIL image.

    Returns:
        float32 HxW array, normalized to [0, 1], 1.0 = nearest.
    """
    out = pipe(img_pil)
    # The HF pipeline returns {"predicted_depth": Tensor, "depth": PIL Image}
    # predicted_depth is at the model's native resolution; we want original resolution.
    depth_t = out["predicted_depth"]  # (1, h_model, w_model) or (h_model, w_model)
    if depth_t.ndim == 2:
        depth_t = depth_t.unsqueeze(0).unsqueeze(0)
    elif depth_t.ndim == 3:
        depth_t = depth_t.unsqueeze(0)
    # resize to original image size
    W, H = img_pil.size
    depth_t = torch.nn.functional.interpolate(
        depth_t, size=(H, W), mode="bicubic", align_corners=False
    )
    depth = depth_t.squeeze().cpu().numpy().astype(np.float32)

    # per-image min-max normalise to [0, 1]
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        # degenerate — uniform depth
        return np.full_like(depth, 0.5, dtype=np.float32)
    depth = (depth - d_min) / (d_max - d_min)

    # Depth Anything outputs INVERSE depth (larger = closer) so after min-max
    # normalisation, 1.0 = nearest. That's our convention, so no flip needed.
    return depth.astype(np.float32)


def process_directory(
    input_dir: str,
    output_dir: str,
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    save_dtype: str = "float16",
    save_preview: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = build_depth_pipeline(model_id, device)

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    preview_dir = out_path / "_previews"
    if save_preview:
        preview_dir.mkdir(exist_ok=True)

    files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in EXTENSIONS)
    print(f"[Depth] {len(files)} images → {out_path}")

    for fpath in tqdm(files, desc="estimating depth"):
        try:
            img = Image.open(fpath).convert("RGB")
            depth = estimate_depth(pipe, img)

            if save_dtype == "float16":
                np.save(str(out_path / f"{fpath.stem}.npy"), depth.astype(np.float16))
            else:
                np.save(str(out_path / f"{fpath.stem}.npy"), depth)

            if save_preview:
                # greyscale preview: 1.0 (near) → white, 0.0 (far) → black
                prev = (depth * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(prev).save(str(preview_dir / f"{fpath.stem}.png"))
        except Exception as e:
            print(f"  [ERROR] {fpath.name}: {e}")

    print(f"[Depth] done. arrays in {out_path}, previews in {preview_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of normal RGB frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store depth .npy files")
    parser.add_argument("--model_id", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace model id. Use Base or Large if Small is noisy.")
    parser.add_argument("--save_dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        save_dtype=args.save_dtype,
        save_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()

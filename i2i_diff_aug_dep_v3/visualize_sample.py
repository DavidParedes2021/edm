#!/usr/bin/env python3
"""
visualize_sample.py — Save per-stage PNG visualizations for one sample.

For a given image stem, dumps the following pure-data PNGs into --output_dir:
    <stem>_normal_rgb.png            Reconstructed normal RGB
    <stem>_normal_L.png              Normal L channel (grayscale, L/100 → [0,255])
    <stem>_normal_L_low.png          Low-pass of normal L  (gaussian, σ=3·max(H,W)/512)
    <stem>_normal_L_high.png         High-pass of normal L (L − L_low), centered at mid-gray
    <stem>_depth.png                 Depth map (grayscale, [0,1] → [0,255])
    <stem>_train_over_L.png          Analytic training-target L (overexposed)
    <stem>_train_under_L.png         Analytic training-target L (underexposed)
    <stem>_infer_over_L.png          Raw diffusion-model L (overexposed) — no postproc
    <stem>_infer_under_L.png         Raw diffusion-model L (underexposed) — no postproc

The inferenced L channels are produced by running DDIM denoise on the source L
+ depth at model resolution and upsampling to the original size. No analytic
floor, no texture reinjection, no chroma — same as diffusion_inference_direct.py
but the L is saved directly instead of being recombined with AB.

USAGE
-----
    python visualize_sample.py \\
        --stem 01c52945-ddf8-47d3-8a53-555dbabdc0a1 \\
        --pairs_dir pair_sample \\
        --over_ckpt path/to/checkpoints/over/best.pt \\
        --under_ckpt path/to/checkpoints/under/best.pt \\
        --config diffusion_config.yaml \\
        --output_dir ./viz_out
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from diffusers import DDIMScheduler

from diffusion_train import (
    build_model, EMAModel, load_config, align_attn_keys_to_model,
)
from exposure_augment import lab_to_rgb


# ─── tiny helpers ──────────────────────────────────────────────────────────

def save_gray_from01(arr01: np.ndarray, path: Path):
    """arr01 in [0, 1] → 8-bit grayscale PNG."""
    a = np.clip(arr01, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def save_gray_from_L(L: np.ndarray, path: Path):
    """L in [0, 100] → 8-bit grayscale PNG."""
    save_gray_from01(L / 100.0, path)


def save_gray_signed(arr: np.ndarray, path: Path, half_range: float = 50.0):
    """Signed map (e.g. high-pass) → grayscale PNG with 0 = mid-gray.
    Values are clipped to ±half_range, then mapped to [0, 255]."""
    a = np.clip(arr, -half_range, half_range)
    save_gray_from01((a + half_range) / (2.0 * half_range), path)


def save_rgb(rgb: np.ndarray, path: Path):
    """rgb uint8 (H,W,3) → PNG."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def resize_2d(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


# ─── one-shot DDIM denoise on a single (source_L, depth) pair ──────────────

@torch.no_grad()
def denoise_once(cfg: dict, ckpt_path: str, domain: str,
                 source_L_full: np.ndarray, depth_full: np.ndarray,
                 device: torch.device, seed: int = None) -> np.ndarray:
    """Return raw model L prediction at full resolution, in [0, 100].

    Identical to diffusion_inference_direct.py's denoise stage but operating
    on a single frame and returning the L map (no AB recombination, no
    postprocessing).
    """
    img_size = cfg["image"]["size"]
    H_o, W_o = source_L_full.shape

    # ── checkpoint + model ──────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg, device)
    if "model" in ckpt:
        ckpt["model"] = align_attn_keys_to_model(ckpt["model"], model)
    if "ema" in ckpt:
        ckpt["ema"] = align_attn_keys_to_model(ckpt["ema"], model)
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(model)
        print(f"[{domain}] loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"[{domain}] loaded raw weights from {ckpt_path}")
    model.eval()

    # ── scheduler ───────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    ddim.set_timesteps(cfg["inference"]["num_inference_steps"], device=device)

    # ── prepare model-resolution conditioning ───────────────────────────
    L_s = resize_2d(source_L_full, img_size)
    L_s = (L_s / 50.0) - 1.0
    d_s = np.clip(resize_2d(depth_full, img_size), 0.0, 1.0) * 2.0 - 1.0

    source_L = torch.from_numpy(L_s).unsqueeze(0).unsqueeze(0).to(device)
    depth_t  = torch.from_numpy(d_s).unsqueeze(0).unsqueeze(0).to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    x = torch.randn_like(source_L)
    for t in ddim.timesteps:
        t_batch = torch.full((1,), t, dtype=torch.long, device=device)
        model_input = torch.cat([x, source_L, depth_t], dim=1)
        pred_noise = model(model_input, t_batch, class_labels=None).sample
        x = ddim.step(pred_noise, t, x).prev_sample

    # denormalise predicted L: [-1, 1] → [0, 100]
    L_pred_small = ((x.cpu().numpy()[0, 0] + 1.0) * 50.0).clip(0, 100)

    # upsample to original resolution (LANCZOS, clip to absorb overshoot)
    L_pred_full = np.array(
        Image.fromarray(L_pred_small.astype(np.float32), mode="F").resize(
            (W_o, H_o), Image.LANCZOS),
        dtype=np.float32,
    ).clip(0.0, 100.0)

    # free GPU memory before next direction
    del model, ckpt, x, source_L, depth_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return L_pred_full


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Save per-stage visualizations for one sample stem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stem", required=True,
                    help="Image stem (filename without extension).")
    ap.add_argument("--pairs_dir", default="pair_sample",
                    help="Directory containing normal/, depth/, "
                         "overexposed/, underexposed/ subdirectories of .npy.")
    ap.add_argument("--over_ckpt", required=True,
                    help="Path to overexposed best.pt.")
    ap.add_argument("--under_ckpt", required=True,
                    help="Path to underexposed best.pt.")
    ap.add_argument("--config", default="diffusion_config.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional seed for reproducible noise.")
    ap.add_argument("--texture_sigma", type=float, default=3.0,
                    help="Gaussian sigma for L low/high decomposition "
                         "(at 512px reference; scaled by max(H,W)/512). "
                         "Matches diffusion_inference.py default.")
    args = ap.parse_args()

    pairs = Path(args.pairs_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    normal_path = pairs / "normal"       / f"{args.stem}.npy"
    depth_path  = pairs / "depth"        / f"{args.stem}.npy"
    over_path   = pairs / "overexposed"  / f"{args.stem}.npy"
    under_path  = pairs / "underexposed" / f"{args.stem}.npy"

    for p in (normal_path, depth_path, over_path, under_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # ── 1. normal LAB → RGB + L channel ────────────────────────────────
    lab = np.load(normal_path).astype(np.float32)        # (H, W, 3) Lab
    L_normal = lab[..., 0]                                # [0, 100]
    rgb_normal = lab_to_rgb(lab)
    save_rgb(rgb_normal,                  out / f"{args.stem}_normal_rgb.png")
    save_gray_from_L(L_normal,            out / f"{args.stem}_normal_L.png")

    # L low/high decomposition — same gaussian split as the inference pipeline
    H_n, W_n = L_normal.shape
    sigma = args.texture_sigma * max(H_n, W_n) / 512.0
    L_low  = gaussian_filter(L_normal, sigma=sigma).astype(np.float32)
    L_high = (L_normal - L_low).astype(np.float32)
    save_gray_from_L(L_low,               out / f"{args.stem}_normal_L_low.png")
    save_gray_signed(L_high,              out / f"{args.stem}_normal_L_high.png")

    # ── 2. depth ───────────────────────────────────────────────────────
    depth = np.load(depth_path).astype(np.float32)        # [0, 1]
    depth = np.clip(depth, 0.0, 1.0)
    save_gray_from01(depth,               out / f"{args.stem}_depth.png")

    # ── 3. analytic training targets ───────────────────────────────────
    L_over_train  = np.load(over_path).astype(np.float32)
    L_under_train = np.load(under_path).astype(np.float32)
    save_gray_from_L(L_over_train,        out / f"{args.stem}_train_over_L.png")
    save_gray_from_L(L_under_train,       out / f"{args.stem}_train_under_L.png")

    # ── 4. raw diffusion-model L for each direction ────────────────────
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device = {device}")

    # depth_full may have a different shape than L_normal — resize to match
    if depth.shape != L_normal.shape:
        depth_for_model = np.array(
            Image.fromarray(depth, mode="F").resize(
                (L_normal.shape[1], L_normal.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
    else:
        depth_for_model = depth

    L_over_pred = denoise_once(
        cfg, args.over_ckpt, "overexposed",
        L_normal, depth_for_model, device, seed=args.seed,
    )
    save_gray_from_L(L_over_pred,         out / f"{args.stem}_infer_over_L.png")

    L_under_pred = denoise_once(
        cfg, args.under_ckpt, "underexposed",
        L_normal, depth_for_model, device, seed=args.seed,
    )
    save_gray_from_L(L_under_pred,        out / f"{args.stem}_infer_under_L.png")

    print(f"\n[viz] wrote 9 PNGs to {out}")


if __name__ == "__main__":
    main()

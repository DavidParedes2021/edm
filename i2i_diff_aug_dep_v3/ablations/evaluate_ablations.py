#!/usr/bin/env python3
"""
ablations/evaluate_ablations.py — Single-checkpoint metric runner.

Given one trained ablation variant (BASELINE / NO_DEPTH / LAB_FULL /
MSE_ONLY / NO_L1 / NO_SOBEL × one direction), this script:

  1. Loads the checkpoint.
  2. Runs MINIMAL DDIM inference (no analytic floor, no focal blend, no
     chroma attenuation, no localisation) on a held-out test split.
     The output is exactly what the model produces, so cross-variant
     differences reflect the model itself rather than a depth-dependent
     post-process step that biases the baseline.
  3. For each (source RGB, GT target_L) pair, computes the
     reference-based + hypothesis-targeted metrics from ablations.metrics.
  4. Saves all generated RGBs to disk so the same folder can be used as
     the input to fid_from_folders() against the unpaired real-domain set.
  5. Writes per-image metrics (JSONL) and an aggregate summary (JSON).

The script is deliberately self-contained: it does not call the full
inference pipeline so it can run identically against all 4 channel
configurations (L-only ± depth, LAB-full ± depth).

USAGE
    python -m ablations.evaluate_ablations \\
        --config       ablations/configs/under/baseline.yaml \\
        --checkpoint   ./outputs/ablations/under/BASELINE/best.pt \\
        --test_normal  ./data/test_normal/under \\
        --test_pairs   ./data/test_pairs                          # for PSNR/SSIM
        --real_dir     ./data/real_under                          # for FID/KID
        --output_dir   ./outputs/ablations/under/BASELINE/eval
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import DDIMScheduler
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# We reuse the model-builder + checkpoint translator from the main training
# script so the ablation eval path can never drift from the way the model is
# actually built at training time.
from diffusion_train import build_model, EMAModel, align_attn_keys_to_model, _channel_counts
from exposure_augment import rgb_to_lab, lab_to_rgb
from ablations import metrics as M


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _list_images(root: str | os.PathLike) -> list:
    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out = []
    for ext in EXTS:
        out.extend(sorted(Path(root).glob(f"*{ext}")))
        out.extend(sorted(Path(root).glob(f"*{ext.upper()}")))
    return out


def _resize_arr(arr: np.ndarray, hw: tuple) -> np.ndarray:
    """Resize a 2-D float array (no clipping) via PIL Lanczos."""
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((hw[1], hw[0]), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal inference — runs DDIM without any depth-dependent post-processing
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_one(model, ddim, cfg, device, rgb_orig: np.ndarray,
              depth_full: np.ndarray | None,
              use_depth: bool, target_mode: str,
              image_size: int) -> dict:
    """Run a single forward pass and return everything we need to compute
    metrics: the predicted L (and AB) at original resolution + the source
    L,AB,depth pieces for cross-checks."""
    H, W = rgb_orig.shape[:2]
    lab_src = rgb_to_lab(rgb_orig)
    L_src = lab_src[..., 0]
    A_src = lab_src[..., 1]
    B_src = lab_src[..., 2]

    L_small = _resize_arr(L_src, (image_size, image_size))
    A_small = _resize_arr(A_src, (image_size, image_size))
    B_small = _resize_arr(B_src, (image_size, image_size))

    # Depth at model resolution (used only if use_depth=True; otherwise
    # the model never sees it so we still build a placeholder to feed the
    # dataset-style API but it gets discarded by the concat below).
    if depth_full is None:
        depth_small = np.full((image_size, image_size), 0.5, dtype=np.float32)
    else:
        depth_small = np.clip(_resize_arr(depth_full, (image_size, image_size)),
                              0.0, 1.0)

    # Build conditioning tensors (normalised to [-1, 1]).
    def _tt(arr): return torch.from_numpy(arr).to(device)
    L_small_n = _tt((L_small / 50.0 - 1.0)[None, None])
    if target_mode == "lab":
        A_small_n = _tt((A_small / 128.0)[None, None])
        B_small_n = _tt((B_small / 128.0)[None, None])
        source_x = torch.cat([L_small_n, A_small_n, B_small_n], dim=1)  # (1,3,H,W)
    else:
        source_x = L_small_n  # (1,1,H,W)
    depth_t = _tt((depth_small * 2.0 - 1.0)[None, None]) if use_depth else None

    # DDIM denoising in target-channel space
    x = torch.randn_like(source_x)
    for t in ddim.timesteps:
        if use_depth:
            inp = torch.cat([x, source_x, depth_t], dim=1)
        else:
            inp = torch.cat([x, source_x], dim=1)
        pred = model(inp, torch.full((1,), int(t), dtype=torch.long, device=device),
                     class_labels=None).sample
        x = ddim.step(pred, t, x).prev_sample

    # Unpack the model output and resize back to original resolution.
    x_np = x.cpu().numpy()[0]  # (C, H, W)
    if target_mode == "l":
        L_pred_small = (x_np[0] + 1.0) * 50.0
        L_pred_full = _resize_arr(L_pred_small, (H, W))
        L_pred_full = np.clip(L_pred_full, 0.0, 100.0)
        # Texture reinjection (consistent with the production inference) —
        # cheap, no depth dependence, applied identically to all L-mode runs.
        sigma = 3.0 * max(H, W) / 512.0
        L_high = L_src - gaussian_filter(L_src.astype(np.float32), sigma=sigma)
        L_low = gaussian_filter(L_pred_full, sigma=sigma)
        L_out = np.clip(L_low + L_high, 0.0, 100.0).astype(np.float32)
        A_out = A_src.astype(np.float32)
        B_out = B_src.astype(np.float32)
    else:
        L_pred_small = (x_np[0] + 1.0) * 50.0
        A_pred_small = x_np[1] * 128.0
        B_pred_small = x_np[2] * 128.0
        L_out = np.clip(_resize_arr(L_pred_small, (H, W)), 0.0, 100.0).astype(np.float32)
        A_out = _resize_arr(A_pred_small, (H, W)).astype(np.float32)
        B_out = _resize_arr(B_pred_small, (H, W)).astype(np.float32)

    lab_out = np.stack([L_out, A_out, B_out], axis=-1)
    rgb_out = lab_to_rgb(lab_out)

    return {
        "rgb_out": rgb_out,
        "L_src": L_src.astype(np.float32),
        "A_src": A_src.astype(np.float32),
        "B_src": B_src.astype(np.float32),
        "L_out": L_out,
        "A_out": A_out,
        "B_out": B_out,
        "depth_full": depth_full,
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_test_list(test_normal_dir: str, test_pairs_dir: str | None,
                    domain: str, test_depth_dir: str | None) -> list:
    """For each normal-frame in the test set, locate the matching GT
    target_L (under <test_pairs_dir>/<domain>/<stem>.npy if present) and
    matching depth map (under <test_depth_dir>/<stem>.npy if present)."""
    items = []
    for p in _list_images(test_normal_dir):
        stem = p.stem
        gt_path = None
        if test_pairs_dir:
            cand = Path(test_pairs_dir) / domain / f"{stem}.npy"
            cand2 = Path(test_pairs_dir) / domain / f"{stem}_v0.npy"
            if cand.exists():
                gt_path = cand
            elif cand2.exists():
                gt_path = cand2
        d_path = None
        if test_depth_dir:
            dc = Path(test_depth_dir) / f"{stem}.npy"
            if dc.exists():
                d_path = dc
        items.append({"rgb": p, "gt_L": gt_path, "depth": d_path, "stem": stem})
    return items


def per_image_metrics(out: dict, gt_L: np.ndarray | None,
                      depth_full: np.ndarray | None,
                      domain: str,
                      rgb_src: np.ndarray | None = None) -> dict:
    """Compute every metric we can given what's available for this image."""
    r = {}
    L_src = out["L_src"]
    L_out = out["L_out"]

    # Source-referenced preservation (matches compute_paired_metrics.py):
    # SSIM/PSNR between the generated exposed frame and the original normal
    # frame. High = the exposure edit preserved texture/structure. This is the
    # *opposite* reference from psnr_L/ssim_L below (which compare to the GT
    # target); a structure-preserving edit scores high here and modestly there.
    if rgb_src is not None:
        pred_rgb = out["rgb_out"]
        if pred_rgb.shape[:2] != rgb_src.shape[:2]:
            pred_rgb = np.array(Image.fromarray(pred_rgb).resize(
                (rgb_src.shape[1], rgb_src.shape[0]), Image.LANCZOS))
        r["ssim_src"] = M.ssim_rgb(pred_rgb, rgb_src, data_range=255.0)
        r["psnr_src"] = M.psnr(pred_rgb, rgb_src, data_range=255.0)

    # Reference-based (need GT)
    if gt_L is not None:
        r["psnr_L"] = M.psnr(L_out, gt_L, data_range=100.0)
        r["ssim_L"] = M.ssim(L_out, gt_L, data_range=100.0)
        # LPIPS uses the RGB reconstruction. For the GT RGB we hold AB
        # fixed at source AB (which is what the synthetic pipeline does)
        # and replace only L. Apples-to-apples between the variants.
        gt_lab = np.stack([gt_L, out["A_src"], out["B_src"]], axis=-1)
        gt_rgb = lab_to_rgb(gt_lab)
        r["lpips_rgb"] = M.lpips_rgb(out["rgb_out"], gt_rgb)

    # Hypothesis-targeted
    if depth_full is not None:
        # Effect strength inside the depth-expected mask
        if domain == "underexposed":
            mask = np.power(np.clip(1.0 - depth_full, 0.0, 1.0), 2.0)
        else:
            mask = np.power(np.clip(depth_full, 0.0, 1.0), 2.0)
        r["delta_L_mask"] = M.delta_L_mask(L_src, L_out, mask)
        # Placement quality
        r["depth_mask_pearson"] = M.depth_mask_pearson(
            L_src, L_out, depth_full, mode=domain, gamma=2.0,
        )

    # Color preservation (matters for A2; identically zero for L-only by construction)
    r["delta_E00_AB"] = M.delta_E00_AB(
        out["A_src"], out["B_src"], out["A_out"], out["B_out"], L_src,
    )

    # Texture preservation
    r["hf_pearson"] = M.hf_pearson(L_src, L_out, sigma=3.0)
    r["sobel_l1"] = M.sobel_l1(L_src, L_out)

    # Effect strength / over-aggression (all variants)
    black_frac, bright_frac = M.extreme_fraction(L_out)
    r["black_frac"] = black_frac
    r["bright_frac"] = bright_frac
    r["mean_delta_L"] = M.mean_delta_L(L_src, L_out)

    return r


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_normal", required=True,
                    help="Folder of held-out normal RGB frames.")
    ap.add_argument("--test_pairs", default=None,
                    help="Optional. <root>/<domain>/<stem>.npy = GT target_L. "
                         "Needed for PSNR/SSIM/LPIPS.")
    ap.add_argument("--test_depth", default=None,
                    help="Optional folder of precomputed depth .npy files for "
                         "the test split. Needed for ΔL_mask, depth-mask Pearson.")
    ap.add_argument("--real_dir", default=None,
                    help="Optional folder of REAL over/under endoscopy frames. "
                         "Triggers FID/KID computation.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--domain", default=None,
                    choices=["overexposed", "underexposed"])
    ap.add_argument("--variant_name", default=None,
                    help="Label written into the summary JSON; defaults to "
                         "the config-file stem.")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device = {device}")

    domain = (args.domain or cfg.get("domain") or "").strip().lower()
    if domain not in ("overexposed", "underexposed"):
        raise SystemExit("--domain must be set (or in config).")
    print(f"[Eval] domain = {domain}")

    use_depth = bool(cfg["model"].get("use_depth", True))
    target_mode = str(cfg["model"].get("target_mode", "l")).lower()
    in_ch, out_ch = _channel_counts(cfg)
    print(f"[Eval] use_depth={use_depth} target_mode={target_mode} "
          f"in_ch={in_ch} out_ch={out_ch}")

    # ── model ────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        ckpt["model"] = align_attn_keys_to_model(ckpt["model"], model)
    if "ema" in ckpt:
        ckpt["ema"] = align_attn_keys_to_model(ckpt["ema"], model)
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(model)
        print("[Eval] loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # ── scheduler ────────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    ddim.set_timesteps(cfg["inference"]["num_inference_steps"], device=device)

    # ── test set ─────────────────────────────────────────────────────────
    items = build_test_list(args.test_normal, args.test_pairs, domain,
                            args.test_depth)
    if args.max_images:
        items = items[: args.max_images]
    print(f"[Eval] {len(items)} test images "
          f"(with GT: {sum(1 for it in items if it['gt_L']) }, "
          f"with depth: {sum(1 for it in items if it['depth']) })")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_image.jsonl"
    summary_path = out_dir / "summary.json"

    # ── per-image pass ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    with open(jsonl_path, "w") as fp:
        for it in tqdm(items, desc=f"Eval {domain}"):
            rgb = np.array(Image.open(it["rgb"]).convert("RGB"))
            depth_full = (np.load(it["depth"]).astype(np.float32)
                          if it["depth"] is not None else None)
            out = infer_one(
                model, ddim, cfg, device, rgb,
                depth_full=depth_full,
                use_depth=use_depth, target_mode=target_mode,
                image_size=cfg["image"]["size"],
            )
            # save generated RGB for FID/KID
            Image.fromarray(out["rgb_out"]).save(str(gen_dir / f"{it['stem']}.png"))

            gt_L = None
            if it["gt_L"] is not None:
                gt_L = np.load(it["gt_L"]).astype(np.float32)
                gt_L = _resize_arr(gt_L, rgb.shape[:2])  # align resolutions

            r = per_image_metrics(out, gt_L, depth_full, domain, rgb_src=rgb)
            r["stem"] = it["stem"]
            fp.write(json.dumps(r) + "\n")
            rows.append(r)

    # ── aggregate ────────────────────────────────────────────────────────
    summary = {
        "variant": args.variant_name or Path(args.config).stem,
        "domain": domain,
        "use_depth": use_depth,
        "target_mode": target_mode,
        "n_images": len(rows),
        "n_with_gt": sum(1 for r in rows if "psnr_L" in r),
        "n_with_depth": sum(1 for r in rows if "depth_mask_pearson" in r),
    }
    metric_keys = ["ssim_src", "psnr_src",
                   "psnr_L", "ssim_L", "lpips_rgb",
                   "delta_L_mask", "depth_mask_pearson",
                   "delta_E00_AB", "hf_pearson", "sobel_l1",
                   "black_frac", "bright_frac", "mean_delta_L"]
    for k in metric_keys:
        vals = [r[k] for r in rows if k in r and not (isinstance(r[k], float)
                                                      and (np.isnan(r[k]) or np.isinf(r[k])))]
        if vals:
            summary[k + "_mean"] = float(np.mean(vals))
            summary[k + "_std"] = float(np.std(vals))

    # ── FID / KID against the unpaired real set ───────────────────────────
    if args.real_dir:
        print(f"[Eval] FID/KID vs {args.real_dir} …")
        fid = M.fid_from_folders(str(gen_dir), args.real_dir)
        summary.update({f"fid_{k}": v for k, v in fid.items()})

    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[Eval] summary → {summary_path}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k:>28s} = {v:.4f}")
        else:
            print(f"    {k:>28s} = {v}")


if __name__ == "__main__":
    main()

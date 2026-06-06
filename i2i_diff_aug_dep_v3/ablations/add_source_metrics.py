#!/usr/bin/env python3
"""
ablations/add_source_metrics.py — Post-hoc source-referenced SSIM/PSNR.

Computes SSIM/PSNR between each variant's already-generated frames and the
ORIGINAL normal source frames (exactly like compute_paired_metrics.py), then
merges `ssim_src_mean/std` and `psnr_src_mean/std` into each variant's
existing eval/summary.json.

No GPU, no model, no re-inference — it reuses the PNGs that
evaluate_ablations.py already saved under
    <root>/<domain>/<VARIANT>/eval/generated/<stem>.png
and matches them by filename stem to the normal frames in --test_normal.

After running this, regenerate the report:
    python -m ablations.aggregate_report --root <root> \\
        --out_md ablations/RESULTS.md --out_csv ablations/RESULTS.csv

USAGE
-----
    python -m ablations.add_source_metrics \\
        --root         ./outputs/ablations \\
        --test_normal  ./outputs/ablations/data/test_normal_subset
    # optional per-direction source dirs (override --test_normal):
    #   --test_normal_under DIR  --test_normal_over DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:                      # works as `python -m ablations.add_source_metrics`
    from ablations import metrics as M
except ImportError:       # ...and as `python ablations/add_source_metrics.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ablations import metrics as M

# Prefer the pipeline's exact L (LAB) for the luminance-based metrics; fall back
# to Rec.709 luma if exposure_augment isn't importable (keeps this script
# dependency-light and runnable from anywhere).
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exposure_augment import rgb_to_lab as _rgb_to_lab

    def _L_of(rgb):
        return _rgb_to_lab(rgb)[..., 0].astype(np.float32)
except Exception:
    def _L_of(rgb):
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return (Y.astype(np.float32) / 255.0) * 100.0

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VARIANT_ORDER = ["BASELINE", "NO_DEPTH", "LAB_FULL", "NO_SOBEL"]


def index_by_stem(folder: Path) -> dict:
    out = {}
    if folder and folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                out.setdefault(p.stem, p)
    return out


def load_rgb(path: Path, target_hw=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_hw is not None and (img.height, img.width) != target_hw:
        img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    return np.array(img)


def score_variant(gen_dir: Path, normal_idx: dict) -> dict | None:
    gen_idx = index_by_stem(gen_dir)
    stems = sorted(set(gen_idx) & set(normal_idx))
    if not stems:
        return None
    ssims, psnrs, blacks, brights, dls = [], [], [], [], []
    for s in stems:
        ref = load_rgb(normal_idx[s])                       # normal source
        gen = load_rgb(gen_idx[s], target_hw=ref.shape[:2])  # generated exposed
        ssims.append(M.ssim_rgb(gen, ref, data_range=255.0))
        psnrs.append(M.psnr(gen, ref, data_range=255.0))
        L_gen, L_ref = _L_of(gen), _L_of(ref)
        bf, wf = M.extreme_fraction(L_gen)
        blacks.append(bf)
        brights.append(wf)
        dls.append(M.mean_delta_L(L_ref, L_gen))
    return {
        "n_src_pairs": len(stems),
        "ssim_src_mean": float(np.mean(ssims)),
        "ssim_src_std": float(np.std(ssims)),
        "psnr_src_mean": float(np.mean(psnrs)),
        "psnr_src_std": float(np.std(psnrs)),
        "black_frac_mean": float(np.mean(blacks)),
        "bright_frac_mean": float(np.mean(brights)),
        "mean_delta_L_mean": float(np.mean(dls)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./outputs/ablations")
    ap.add_argument("--test_normal", default=None,
                    help="Folder of normal source frames (used for both directions "
                         "unless overridden per-direction).")
    ap.add_argument("--test_normal_under", default=None)
    ap.add_argument("--test_normal_over", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    normal_dirs = {
        "under": Path(args.test_normal_under or args.test_normal or ""),
        "over":  Path(args.test_normal_over or args.test_normal or ""),
    }
    for d, p in normal_dirs.items():
        if not p or not p.is_dir():
            raise SystemExit(f"[{d}] normal source dir not found: {p!r} "
                             f"(pass --test_normal or --test_normal_{d})")

    normal_idx = {d: index_by_stem(p) for d, p in normal_dirs.items()}
    for d, idx in normal_idx.items():
        print(f"[{d}] {len(idx)} normal source frames in {normal_dirs[d]}")

    for domain in ("under", "over"):
        for v in VARIANT_ORDER:
            gen_dir = root / domain / v / "eval" / "generated"
            summ_path = root / domain / v / "eval" / "summary.json"
            if not summ_path.exists():
                print(f"[skip] {domain}/{v}: no summary.json")
                continue
            res = score_variant(gen_dir, normal_idx[domain])
            if res is None:
                print(f"[skip] {domain}/{v}: no matching generated/normal stems")
                continue
            with open(summ_path) as f:
                summary = json.load(f)
            summary.update(res)
            with open(summ_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[ok] {domain}/{v}: SSIM_src={res['ssim_src_mean']:.4f} "
                  f"PSNR_src={res['psnr_src_mean']:.2f} (n={res['n_src_pairs']}) "
                  f"-> {summ_path}")

    print("\nNow re-run the aggregator to refresh RESULTS.md / RESULTS.csv:")
    print(f"  python -m ablations.aggregate_report --root {root} "
          f"--out_md ablations/RESULTS.md --out_csv ablations/RESULTS.csv")


if __name__ == "__main__":
    main()

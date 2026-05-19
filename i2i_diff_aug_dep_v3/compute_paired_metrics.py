#!/usr/bin/env python3
"""
compute_paired_metrics.py — SSIM / PSNR / LPIPS between normal frames and
their synthetic overexposed / underexposed counterparts.

Pairs are matched by filename stem. Each generated image is resized to its
reference's resolution before SSIM/PSNR; LPIPS is computed at 256x256.

USAGE
-----
    python compute_paired_metrics.py \
        --normal path/to/normal \
        --over   path/to/synthetic_overexposed \
        --under  path/to/synthetic_underexposed \
        --output_dir ./paired_metrics_run

Either --over or --under (or both) can be provided.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMG_EXTS)


def common_stems(dir_a, dir_b):
    a = {p.stem: p for p in list_images(dir_a)}
    b = {p.stem: p for p in list_images(dir_b)}
    keys = sorted(set(a) & set(b))
    return [(a[k], b[k]) for k in keys]


def load_rgb(path, target_hw=None):
    img = Image.open(path).convert("RGB")
    if target_hw is not None:
        img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    return np.array(img)


def ssim_psnr(gen, ref):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    s = structural_similarity(gen, ref, channel_axis=-1, data_range=255)
    p = peak_signal_noise_ratio(ref, gen, data_range=255)
    return float(s), float(p)


def load_lpips(device):
    import lpips
    print("[metrics] loading LPIPS (AlexNet)...")
    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_batch(gens, refs, lpips_model, device):
    def to_t(rgb):
        return (torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1.0
                ).unsqueeze(0).to(device)
    out = []
    with torch.no_grad():
        for g, r in zip(gens, refs):
            out.append(float(lpips_model(to_t(g), to_t(r)).item()))
    return out


def paired_metrics(normal_dir, gen_dir, csv_path, lpips_model, device, label):
    pairs = common_stems(normal_dir, gen_dir)
    if not pairs:
        print(f"[{label}] no matching stems between {normal_dir} and {gen_dir}")
        return {"n_pairs": 0}

    rows, g256, r256 = [], [], []
    for np_path, gp in pairs:
        ref = load_rgb(np_path)
        gen = load_rgb(gp, target_hw=ref.shape[:2])
        s, p = ssim_psnr(gen, ref)
        rows.append({"stem": gp.stem, "ssim": s, "psnr": p})
        g256.append(np.array(Image.fromarray(gen).resize((256, 256), Image.LANCZOS)))
        r256.append(np.array(Image.fromarray(ref).resize((256, 256), Image.LANCZOS)))

    print(f"[{label}] {len(rows)} matched pairs; computing LPIPS...")
    lp = lpips_batch(g256, r256, lpips_model, device)
    for row, v in zip(rows, lp):
        row["lpips"] = v

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "ssim", "psnr", "lpips"])
        w.writeheader()
        w.writerows(rows)

    arr = lambda k: np.array([r[k] for r in rows])
    summary = {
        "n_pairs":    len(rows),
        "ssim_mean":  float(arr("ssim").mean()),
        "ssim_std":   float(arr("ssim").std()),
        "psnr_mean":  float(arr("psnr").mean()),
        "psnr_std":   float(arr("psnr").std()),
        "lpips_mean": float(arr("lpips").mean()),
        "lpips_std":  float(arr("lpips").std()),
        "csv":        str(csv_path),
    }
    print(f"[{label}] SSIM  = {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"[{label}] PSNR  = {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")
    print(f"[{label}] LPIPS = {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Paired SSIM/PSNR/LPIPS: normal vs synthetic over/under.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--normal", required=True, help="Directory of normal (reference) images.")
    ap.add_argument("--over",   help="Directory of synthetic overexposed images.")
    ap.add_argument("--under",  help="Directory of synthetic underexposed images.")
    ap.add_argument("--output_dir", required=True, help="Where to write per-pair CSVs and summary.")
    args = ap.parse_args()

    if not (args.over or args.under):
        sys.exit("[metrics] provide at least one of --over / --under.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[metrics] device = {device}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    lpips_model = load_lpips(device)

    results = {}
    if args.over:
        results["over_vs_normal"] = paired_metrics(
            args.normal, args.over,
            out_root / "over_vs_normal.csv",
            lpips_model, device, label="over",
        )
    if args.under:
        results["under_vs_normal"] = paired_metrics(
            args.normal, args.under,
            out_root / "under_vs_normal.csv",
            lpips_model, device, label="under",
        )

    summary_path = out_root / "summary.csv"
    fields = ["pair", "n_pairs",
              "ssim_mean", "ssim_std",
              "psnr_mean", "psnr_std",
              "lpips_mean", "lpips_std"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in results.items():
            if r.get("n_pairs", 0):
                w.writerow({"pair": name, **{k: r[k] for k in fields[1:]}})

    print(f"\n[metrics] summary → {summary_path}")


if __name__ == "__main__":
    main()

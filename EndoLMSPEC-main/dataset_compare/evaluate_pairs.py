#!/usr/bin/env python3
"""
evaluate_pairs.py — run a trained EndoLMSPEC model on a paired test set and
report full-reference image-quality metrics (SSIM, PSNR, and LPIPS if the
`lpips` package is installed).

This is a clean, wandb-free replacement for test_model.py, built for the
dataset comparison. It expects a test folder produced by prepare_data.py:

    <test_dir>/
        exp_images/   (corrupted input)
        gt_images/    (normal GT, same filenames)

It writes:
  * a per-image CSV  (<out_csv>)
  * one aggregate row appended to <summary_csv> tagged with --train_tag /
    --test_tag, so the cross-dataset matrix can be assembled afterwards.

Example
-------
python evaluate_pairs.py \
    --model      ./checkpoints/mine_over/main_net/model_256.pth \
    --test_dir   ./compare_data/endo4ie_over/test \
    --train_tag  mine_over --test_tag endo4ie_over \
    --out_csv    ./results/mine_over__on__endo4ie_over.csv \
    --summary_csv ./results/summary.csv --gpu 0
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# make the EndoLMSPEC package importable when run from this subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generator import Generator  # noqa: E402

try:
    import lpips as lpips_pkg
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def load_model(model_path, device):
    net = Generator(n_channels=3, device=device, bilinear=False)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def to_tensor(pil_img, size, device):
    if size > 0:
        pil_img = pil_img.resize((size, size), Image.BICUBIC)
    t = T.ToTensor()(pil_img).unsqueeze(0).to(device=device, dtype=torch.float32)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_dir", required=True, help="folder with exp_images/ and gt_images/")
    ap.add_argument("--train_tag", required=True, help="which dataset the model was trained on")
    ap.add_argument("--test_tag", required=True, help="which dataset this test set comes from")
    ap.add_argument("--out_csv", required=True, help="per-image CSV output path")
    ap.add_argument("--summary_csv", required=True, help="aggregate CSV (appended to)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    net = load_model(args.model, device)

    lpips_model = None
    if _HAS_LPIPS:
        lpips_model = lpips_pkg.LPIPS(net="alex").to(device).eval()

    exp_dir = Path(args.test_dir) / "exp_images"
    gt_dir = Path(args.test_dir) / "gt_images"
    files = sorted(p.name for p in exp_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No images in {exp_dir}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    ssims, psnrs, lpipss = [], [], []

    with torch.no_grad():
        for fname in files:
            gt_path = gt_dir / fname
            if not gt_path.exists():
                print(f"[skip] no GT for {fname}")
                continue

            exp_t = to_tensor(Image.open(exp_dir / fname).convert("RGB"), args.size, device)
            _, out = net(exp_t)
            pred = out["subnet_16"].clamp(0.0, 1.0)          # (1,3,H,W)
            gt_t = to_tensor(Image.open(gt_path).convert("RGB"), args.size, device)

            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
            gt_np = gt_t[0].permute(1, 2, 0).cpu().numpy()

            s = float(ssim_fn(gt_np, pred_np, channel_axis=2, data_range=1.0))
            p = float(psnr_fn(gt_np, pred_np, data_range=1.0))
            l = ""
            if lpips_model is not None:
                # lpips expects [-1,1]
                lp = lpips_model(pred * 2 - 1, gt_t * 2 - 1)
                l = float(lp.item())
                lpipss.append(l)

            ssims.append(s)
            psnrs.append(p)
            rows.append({"file": fname, "ssim": s, "psnr": p, "lpips": l})

    # per-image CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "ssim", "psnr", "lpips"])
        w.writeheader()
        w.writerows(rows)

    def ms(arr):
        a = np.array(arr, dtype=np.float64)
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))

    ssim_m, ssim_sd = ms(ssims)
    psnr_m, psnr_sd = ms(psnrs)
    lpips_m, lpips_sd = ms(lpipss) if lpipss else ("", "")

    print(f"\n=== {args.train_tag}  ->  tested on  {args.test_tag}  (n={len(rows)}) ===")
    print(f"SSIM  {ssim_m:.4f} ± {ssim_sd:.4f}")
    print(f"PSNR  {psnr_m:.2f} ± {psnr_sd:.2f} dB")
    if lpipss:
        print(f"LPIPS {lpips_m:.4f} ± {lpips_sd:.4f}  (lower is better)")
    else:
        print("LPIPS  (skipped: `pip install lpips` to enable)")

    # append aggregate row to the summary
    summary = Path(args.summary_csv)
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["train_tag", "test_tag", "n",
                        "ssim_mean", "ssim_std", "psnr_mean", "psnr_std",
                        "lpips_mean", "lpips_std", "model"])
        w.writerow([args.train_tag, args.test_tag, len(rows),
                    f"{ssim_m:.4f}", f"{ssim_sd:.4f}", f"{psnr_m:.2f}", f"{psnr_sd:.2f}",
                    (f"{lpips_m:.4f}" if lpipss else ""), (f"{lpips_sd:.4f}" if lpipss else ""),
                    os.path.abspath(args.model)])
    print(f"appended summary row -> {summary}")


if __name__ == "__main__":
    main()

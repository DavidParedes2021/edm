#!/usr/bin/env python3
"""
ablations/aggregate_report.py — Walk all `summary.json` files written by
evaluate_ablations.py and emit one markdown + one CSV comparison table.

Layout assumed:
    <root>/<under|over>/<VARIANT>/eval/summary.json

Usage:
    python -m ablations.aggregate_report \\
        --root ./outputs/ablations \\
        --out_md  ablations/RESULTS.md \\
        --out_csv ablations/RESULTS.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Order shown in the report — also picks which columns the markdown shows.
METRIC_ORDER = [
    ("psnr_L_mean",            "PSNR↑"),
    ("ssim_L_mean",            "SSIM↑"),
    ("lpips_rgb_mean",         "LPIPS↓"),
    ("fid_fid",                "FID↓"),
    ("fid_kid",                "KID↓"),
    ("delta_L_mask_mean",      "ΔL_mask↑"),
    ("depth_mask_pearson_mean","Depth-r↑"),
    ("delta_E00_AB_mean",      "ΔE_AB↓"),
    ("hf_pearson_mean",        "HF-r↑"),
    ("sobel_l1_mean",          "Sobel↓"),
]

VARIANT_ORDER = ["BASELINE", "NO_DEPTH", "LAB_FULL",
                 "MSE_ONLY", "NO_L1", "NO_SOBEL"]


def collect(root: Path) -> dict:
    """Return {domain: {variant: summary_dict}}."""
    out = {}
    for domain in ("under", "over"):
        out[domain] = {}
        for v in VARIANT_ORDER:
            p = root / domain / v / "eval" / "summary.json"
            if p.exists():
                with open(p) as f:
                    out[domain][v] = json.load(f)
            else:
                out[domain][v] = None
    return out


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def markdown_table(data: dict) -> str:
    lines = ["# Ablation results", ""]
    for domain_short, domain_long in (("under", "Underexposed"),
                                       ("over", "Overexposed")):
        lines.append(f"## {domain_long}")
        lines.append("")
        header = "| Variant | " + " | ".join(label for _, label in METRIC_ORDER) + " |"
        sep    = "|---|" + "|".join("---" for _ in METRIC_ORDER) + "|"
        lines.append(header)
        lines.append(sep)
        for v in VARIANT_ORDER:
            s = data[domain_short].get(v)
            if s is None:
                row = [v] + ["—"] * len(METRIC_ORDER)
            else:
                row = [v] + [fmt(s.get(k)) for k, _ in METRIC_ORDER]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def csv_rows(data: dict) -> list:
    rows = [["domain", "variant"] + [k for k, _ in METRIC_ORDER]]
    for domain in ("under", "over"):
        for v in VARIANT_ORDER:
            s = data[domain].get(v)
            if s is None:
                rows.append([domain, v] + ["" for _ in METRIC_ORDER])
            else:
                rows.append([domain, v] + [s.get(k, "") for k, _ in METRIC_ORDER])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./outputs/ablations")
    ap.add_argument("--out_md",  default="ablations/RESULTS.md")
    ap.add_argument("--out_csv", default="ablations/RESULTS.csv")
    args = ap.parse_args()

    data = collect(Path(args.root))
    md = markdown_table(data)
    with open(args.out_md, "w") as f:
        f.write(md)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows(data))
    print(f"[wrote] {args.out_md}")
    print(f"[wrote] {args.out_csv}")
    print()
    print(md)


if __name__ == "__main__":
    main()

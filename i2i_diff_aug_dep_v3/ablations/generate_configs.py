#!/usr/bin/env python3
"""
ablations/generate_configs.py — Materialise the 12 ablation YAML configs.

Reads the project's `diffusion_config.yaml` as a base, applies the
ablation-specific overrides (model channels, loss weights, post-processing
toggles), and writes one YAML per (variant × direction) into
`ablations/configs/<under|over>/<variant>.yaml`.

Run once at setup:
    python -m ablations.generate_configs --base diffusion_config.yaml \\
        --pairs_baseline    PATH_TO_DEPTH_AWARE_PAIRS \\
        --pairs_no_depth    PATH_TO_CLUSTER_ONLY_PAIRS \\
        --depth_dir         PATH_TO_DEPTH_CACHE \\
        --epochs 150 \\
        --out ablations/configs
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variant table.
#   Each entry is a function that mutates a copy of the base config in-place.
#   The variant name is used as the YAML stem and the variant_name field in
#   downstream metric summaries.
# ─────────────────────────────────────────────────────────────────────────────
def _v_baseline(cfg):
    # Full pipeline — same as production. Just make sure the toggles are
    # explicit so a stale config can't silently change behaviour.
    cfg["model"]["use_depth"] = True
    cfg["model"]["target_mode"] = "l"


def _v_no_depth(cfg):
    # A1 — depth removed from EVERY layer of the pipeline.
    cfg["model"]["use_depth"] = False
    cfg["model"]["target_mode"] = "l"
    cfg["losses"]["depth_grad_weight"] = 0.0
    cfg["focal_blend"]["enable"] = False
    cfg["analytic_floor"]["enable"] = False        # uses depth_aware_augment
    cfg["chroma"]["depth_chroma_boost"] = 0.0      # depth-modulated desat off
    cfg["data"]["pairs_dir"] = "__PAIRS_NO_DEPTH__"  # filled at write-time


def _v_lab_full(cfg):
    # A2 — diffusion learns L AND A,B together.
    cfg["model"]["use_depth"] = True
    cfg["model"]["target_mode"] = "lab"


def _v_no_sobel(cfg):
    cfg["losses"]["edge_weight"] = 0.0


VARIANTS = {
    "BASELINE": _v_baseline,
    "NO_DEPTH": _v_no_depth,
    "LAB_FULL": _v_lab_full,
    "NO_SOBEL": _v_no_sobel,
}


def _make_one(base_cfg: dict, variant: str, domain: str,
              pairs_baseline: str, pairs_no_depth: str,
              depth_dir: str | None, epochs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["domain"] = domain
    cfg["training"]["epochs"] = epochs
    cfg["training"]["disable_samples"] = True   # ablations skip the visual sampler
    # Default pairs/depth — overridden below for NO_DEPTH.
    cfg["data"]["pairs_dir"] = pairs_baseline
    if depth_dir:
        cfg["data"]["depth_dir"] = depth_dir
    VARIANTS[variant](cfg)
    if cfg["data"]["pairs_dir"] == "__PAIRS_NO_DEPTH__":
        cfg["data"]["pairs_dir"] = pairs_no_depth
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="diffusion_config.yaml")
    ap.add_argument("--pairs_baseline", required=True,
                    help="Path to depth-aware pairs (BASELINE/LAB_FULL/loss variants).")
    ap.add_argument("--pairs_no_depth", required=True,
                    help="Path to cluster-only pairs (NO_DEPTH variant).")
    ap.add_argument("--depth_dir", default=None,
                    help="Optional depth cache used by the dataset / inference.")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="ablations/configs")
    args = ap.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)

    out_root = Path(args.out)
    written = []
    for domain_short, domain_long in (("under", "underexposed"), ("over", "overexposed")):
        d = out_root / domain_short
        d.mkdir(parents=True, exist_ok=True)
        for v in VARIANTS:
            cfg = _make_one(
                base, v, domain_long,
                pairs_baseline=args.pairs_baseline,
                pairs_no_depth=args.pairs_no_depth,
                depth_dir=args.depth_dir,
                epochs=args.epochs,
            )
            # Per-variant output sub-tree so checkpoints don't collide.
            cfg["output"]["checkpoints_dir"] = \
                f"./outputs/ablations/{domain_short}/{v}/checkpoints"
            cfg["output"]["samples_dir"] = \
                f"./outputs/ablations/{domain_short}/{v}/samples"
            cfg["output"]["root"] = \
                f"./outputs/ablations/{domain_short}/{v}"
            p = d / f"{v.lower()}.yaml"
            with open(p, "w") as fp:
                yaml.safe_dump(cfg, fp, sort_keys=False)
            written.append(p)
            print(f"[wrote] {p}")
    print(f"[done] {len(written)} configs written under {out_root}")


if __name__ == "__main__":
    main()

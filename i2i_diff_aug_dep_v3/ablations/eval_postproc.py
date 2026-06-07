#!/usr/bin/env python3
"""
ablations/eval_postproc.py — Metrics WITH the full production post-processing.

The standard ablation evaluator (evaluate_ablations.py) runs MINIMAL inference
(DDIM + texture reinjection) so all four variants are comparable. This script
instead runs the REAL production pipeline (analytic floor, texture gate, focal
blend, chroma attenuation) by calling diffusion_inference.run_inference, then
scores the result and merges the numbers into each variant's summary.json with
a `_pp` suffix (post-processed).

Only the variants whose architecture the production pipeline supports are
"applicable": it builds cat([x, source_L, depth]) and assumes an L-only target,
so it works for the depth + L-mode variants only:

        APPLICABLE = BASELINE, NO_SOBEL

NO_DEPTH (no depth channel) and LAB_FULL (predicts A,B) are incompatible and are
skipped automatically.

Adds these per variant (post-processed):
    ssim_src_pp, psnr_src_pp   — vs the normal source (texture preservation)
    ssim_L_pp,  psnr_L_pp      — vs the GT target L (exposure fidelity)
    lpips_pp                   — vs the GT target RGB
    fid_pp, kid_pp             — vs the unpaired real over/under set
    black_frac_pp, bright_frac_pp, mean_delta_L_pp

USAGE
-----
  set -a; . ablations/scripts/ablation.env; set +a
  python -m ablations.eval_postproc \
      --root        "$ABL_ROOT" \
      --test_normal "$TEST_NORMAL_SUBSET" \
      --test_pairs  "$TEST_PAIRS" \
      --test_depth  "$DEPTH_TEST_DIR" \
      --real_under  "$REAL_UNDER_DIR" \
      --real_over   "$REAL_OVER_DIR" \
      --hadepth_repo "$HADEPTH_REPO" --hadepth_ckpt "$HADEPTH_CKPT"

Then refresh the report:
  python -m ablations.aggregate_report --root "$ABL_ROOT" \
      --out_md ablations/RESULTS.md --out_csv ablations/RESULTS.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # i2i_diff_aug_dep_v3
from diffusion_inference import run_inference          # noqa: E402
from exposure_augment import rgb_to_lab, lab_to_rgb    # noqa: E402
from ablations import metrics as M                     # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
APPLICABLE = ["BASELINE", "NO_SOBEL"]   # depth + L-mode only


def index_by_stem(folder: Path) -> dict:
    out = {}
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                out.setdefault(p.stem, p)
    return out


def load_rgb(path: Path, target_hw=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_hw is not None and (img.height, img.width) != tuple(target_hw):
        img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    return np.array(img)


def resize_L(arr: np.ndarray, hw) -> np.ndarray:
    if arr.shape[:2] == tuple(hw):
        return arr
    return np.array(Image.fromarray(arr.astype(np.float32), mode="F").resize(
        (hw[1], hw[0]), Image.LANCZOS), dtype=np.float32)


def checkpoint_candidates(cfg: dict, root: Path, domain_short: str, v: str,
                          domain_long: str, override_root: str | None) -> list:
    """Ordered list of plausible best.pt locations. The config's own
    output.checkpoints_dir is authoritative (it's what the trainer wrote to)."""
    cands = []
    if override_root:
        cands.append(Path(override_root) / domain_short / v / "checkpoints" / domain_long / "best.pt")
    cd = (cfg.get("output", {}) or {}).get("checkpoints_dir")
    if cd:
        cands.append(Path(cd) / domain_long / "best.pt")
        cands.append(Path(cd) / "best.pt")
    cands.append(Path("outputs/ablations") / domain_short / v / "checkpoints" / domain_long / "best.pt")
    cands.append(root / domain_short / v / "checkpoints" / domain_long / "best.pt")
    return cands


def resolve_checkpoint(cfg, root, domain_short, v, domain_long, override_root):
    cands = checkpoint_candidates(cfg, root, domain_short, v, domain_long, override_root)
    for c in cands:
        if c.exists():
            return c, cands
    # last-resort recursive search under the config's output root
    cd = (cfg.get("output", {}) or {}).get("checkpoints_dir")
    for base in [Path(cd) if cd else None, Path("outputs/ablations"), root]:
        if base and base.exists():
            hits = [h for h in base.rglob("best.pt")
                    if v in h.parts and domain_long in str(h)]
            if hits:
                return hits[0], cands
    return None, cands


def resolve_summary(cfg, root, domain_short, v) -> Path:
    """Where to merge the _pp keys — prefer an existing summary.json so it lands
    in the file aggregate_report will read."""
    cand_cfg = Path((cfg.get("output", {}) or {}).get("root", "")) / "eval" / "summary.json" \
        if (cfg.get("output", {}) or {}).get("root") else None
    cand_root = root / domain_short / v / "eval" / "summary.json"
    for c in (cand_cfg, cand_root):
        if c and c.exists():
            return c
    return cand_root   # default: root-based, so `aggregate_report --root` finds it


def find_gt(test_pairs_domain: Path, stem: str):
    for cand in (test_pairs_domain / f"{stem}.npy",
                 test_pairs_domain / f"{stem}_v0.npy"):
        if cand.exists():
            return cand
    return None


def score_folder(gen_dir: Path, normal_idx: dict, test_pairs_domain: Path,
                 real_dir: str | None, max_images: int | None) -> dict:
    gen_idx = index_by_stem(gen_dir)
    stems = sorted(set(gen_idx) & set(normal_idx))
    if max_images:
        stems = stems[:max_images]
    if not stems:
        return {}

    acc = {k: [] for k in ("ssim_src", "psnr_src", "ssim_L", "psnr_L", "lpips",
                           "black_frac", "bright_frac", "mean_delta_L")}
    for s in stems:
        src = load_rgb(normal_idx[s])
        gen = load_rgb(gen_idx[s], target_hw=src.shape[:2])
        acc["ssim_src"].append(M.ssim_rgb(gen, src, data_range=255.0))
        acc["psnr_src"].append(M.psnr(gen, src, data_range=255.0))

        gen_L = rgb_to_lab(gen)[..., 0].astype(np.float32)
        src_lab = rgb_to_lab(src)
        bf, wf = M.extreme_fraction(gen_L)
        acc["black_frac"].append(bf)
        acc["bright_frac"].append(wf)
        acc["mean_delta_L"].append(M.mean_delta_L(src_lab[..., 0], gen_L))

        gt = find_gt(test_pairs_domain, s)
        if gt is not None:
            gt_L = resize_L(np.load(gt).astype(np.float32), gen_L.shape[:2])
            acc["ssim_L"].append(M.ssim(gen_L, gt_L, data_range=100.0))
            acc["psnr_L"].append(M.psnr(gen_L, gt_L, data_range=100.0))
            gt_rgb = lab_to_rgb(np.stack([gt_L, src_lab[..., 1], src_lab[..., 2]], -1))
            acc["lpips"].append(M.lpips_rgb(gen, gt_rgb))

    out = {}
    for k, vals in acc.items():
        v = [x for x in vals if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
        if v:
            out[f"{k}_pp_mean"] = float(np.mean(v))
            out[f"{k}_pp_std"] = float(np.std(v))
    out["n_pp"] = len(stems)

    if real_dir:
        print(f"   [pp] FID/KID vs {real_dir} …")
        fid = M.fid_from_folders(str(gen_dir), real_dir)
        out["fid_pp"] = fid["fid"]
        out["kid_pp"] = fid["kid"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ABL_ROOT (outputs/ablations)")
    ap.add_argument("--configs_dir", default="ablations/configs")
    ap.add_argument("--test_normal", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--test_depth", required=True)
    ap.add_argument("--real_under", default=None)
    ap.add_argument("--real_over", default=None)
    ap.add_argument("--hadepth_repo", default=None)
    ap.add_argument("--hadepth_ckpt", default=None)
    ap.add_argument("--checkpoints_root", default=None,
                    help="optional override for where checkpoints live "
                         "(e.g. ./outputs/ablations). Normally auto-derived "
                         "from each config's output.checkpoints_dir.")
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    configs_dir = Path(args.configs_dir)
    normal_idx = index_by_stem(Path(args.test_normal))
    print(f"[pp] {len(normal_idx)} normal source frames")

    for domain_short, domain_long in (("under", "underexposed"), ("over", "overexposed")):
        real_dir = args.real_under if domain_short == "under" else args.real_over
        for v in APPLICABLE:
            cfg_path = configs_dir / domain_short / f"{v.lower()}.yaml"
            if not cfg_path.exists():
                print(f"[skip] {domain_short}/{v}: config not found at {cfg_path.resolve()}")
                continue
            cfg = yaml.safe_load(open(cfg_path))

            ckpt, cands = resolve_checkpoint(cfg, root, domain_short, v,
                                             domain_long, args.checkpoints_root)
            if ckpt is None:
                print(f"[skip] {domain_short}/{v}: no best.pt found. checked:")
                for c in cands:
                    print(f"          {c}")
                continue
            summ_path = resolve_summary(cfg, root, domain_short, v)
            print(f"[pp] {domain_short}/{v}: ckpt={ckpt}")

            use_depth = bool(cfg["model"].get("use_depth", True))
            target_mode = str(cfg["model"].get("target_mode", "l")).lower()
            if not (use_depth and target_mode == "l"):
                print(f"[skip] {domain_short}/{v}: not applicable to production "
                      f"post-processing (use_depth={use_depth}, target_mode={target_mode})")
                continue

            # point inference at the held-out test frames
            cfg2 = copy.deepcopy(cfg)
            cfg2["data"]["normal_dir"] = args.test_normal

            gen_root = summ_path.parent / "generated_pp"   # beside the summary
            print(f"\n[pp] {domain_short}/{v}: production inference → {gen_root}")
            run_inference(
                cfg2, str(ckpt),
                domain=domain_long,
                output_dir=str(gen_root),
                depth_dir=args.test_depth,
                depth_backend="hadepth",
                hadepth_repo=args.hadepth_repo,
                hadepth_ckpt=args.hadepth_ckpt,
            )
            gen_dir = gen_root / domain_long   # run_inference writes <output_dir>/<domain>/

            res = score_folder(gen_dir, normal_idx,
                               Path(args.test_pairs) / domain_long,
                               real_dir, args.max_images)
            if not res:
                print(f"[warn] {domain_short}/{v}: no scored frames")
                continue

            summary = json.load(open(summ_path)) if summ_path.exists() else {}
            summary.update(res)
            summ_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(summary, open(summ_path, "w"), indent=2)
            print(f"[ok] {domain_short}/{v}: "
                  f"SSIM_src_pp={res.get('ssim_src_pp_mean', float('nan')):.4f} "
                  f"SSIM_L_pp={res.get('ssim_L_pp_mean', float('nan')):.4f} "
                  f"FID_pp={res.get('fid_pp', float('nan')):.1f} "
                  f"black_pp={res.get('black_frac_pp_mean', float('nan'))*100:.0f}% "
                  f"-> {summ_path}")

    print("\nNow refresh the report:")
    print(f"  python -m ablations.aggregate_report --root {root} "
          f"--out_md ablations/RESULTS.md --out_csv ablations/RESULTS.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ablations/check_depth.py — Is the depth signal actually usable?

Diagnoses whether the cached depth maps (HADepth or fallback) are correct and
informative, which decides whether the under/over BASELINE failure is a
DEPTH-DATA problem or a TRAINING problem.

Two checks, no GPU:

  1. HADepth load status — greps the setup logs for the
     "[depth] HADepth loaded: matched N/total keys" line. A low match ratio
     means the checkpoint didn't really load (strict=False) -> garbage depth.

  2. Depth-vs-luminance orientation. In endoscopy the light source is co-located
     with the camera, so illumination falls off with distance: NEAR tissue is
     bright, the FAR cavity is dark. A correct depth map (1=near, 0=far) must
     therefore POSITIVELY correlate with source luminance over visible tissue.

         mean r > +0.2   -> depth is oriented correctly and informative
        -0.2..+0.2       -> depth is uninformative (likely ckpt not loaded)
         mean r < -0.2   -> depth is INVERTED (near/far swapped)

It also writes side-by-side panels [RGB | depth | L | expected far-mask] so you
can eyeball where the under-augmentation *should* land.

USAGE
-----
  python -m ablations.check_depth \
      --depth_dir  "$DEPTH_TEST_DIR" \
      --normal_dir "$TEST_NORMAL_SUBSET" \
      --out_dir    ./ablations/depth_check \
      --setup_log  "$ABL_ROOT/logs/setup_depth_train.log" \
      --max_images 40
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma scaled to [0,100], matching the pipeline's L convention."""
    Y = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    return (Y.astype(np.float32) / 255.0) * 100.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def index_by_stem(folder: Path) -> dict:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            out.setdefault(p.stem, p)
    return out


def gray_to_img(arr01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")


def report_hadepth_log(setup_log: Path):
    if not setup_log or not setup_log.exists():
        print("[hadepth] no setup log given/found — skipping load-status check.")
        return
    text = setup_log.read_text(errors="ignore")
    hits = [ln for ln in text.splitlines()
            if "HADepth loaded" in ln or "backend=" in ln or "[depth]" in ln]
    print(f"[hadepth] relevant lines from {setup_log.name}:")
    for ln in hits[:12]:
        print("   ", ln.strip())
    m = re.search(r"matched\s+(\d+)/(\d+)\s+keys", text)
    if m:
        n, tot = int(m.group(1)), int(m.group(2))
        frac = n / max(tot, 1)
        verdict = "OK" if frac > 0.9 else ("PARTIAL" if frac > 0.5 else "BROKEN")
        print(f"[hadepth] checkpoint match = {n}/{tot} ({frac:.0%})  -> {verdict}")
        if frac <= 0.9:
            print("[hadepth] ** low match ratio: depth maps are likely unreliable. **")
    else:
        print("[hadepth] no 'matched N/total keys' line found "
              "(HADepth may not have run, or log is from another backend).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_dir", required=True, help="folder of depth .npy maps (1=near)")
    ap.add_argument("--normal_dir", required=True, help="matching normal source frames")
    ap.add_argument("--out_dir", default="./ablations/depth_check")
    ap.add_argument("--setup_log", default=None, help="optional depth setup log to grep")
    ap.add_argument("--max_images", type=int, default=40)
    ap.add_argument("--n_panels", type=int, default=8, help="how many visual panels to save")
    ap.add_argument("--vignette_threshold", type=float, default=5.0)
    args = ap.parse_args()

    report_hadepth_log(Path(args.setup_log) if args.setup_log else None)

    depth_idx = index_by_stem(Path(args.depth_dir)) if False else {
        p.stem: p for p in Path(args.depth_dir).rglob("*.npy")
    }
    normal_idx = index_by_stem(Path(args.normal_dir))
    stems = sorted(set(depth_idx) & set(normal_idx))
    if not stems:
        raise SystemExit(f"No matching stems between {args.depth_dir} and {args.normal_dir}")
    stems = stems[: args.max_images]
    print(f"\n[orient] checking {len(stems)} frames (depth vs luminance correlation)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corrs = []
    for i, s in enumerate(stems):
        depth = np.load(depth_idx[s]).astype(np.float32)
        rgb = np.array(Image.open(normal_idx[s]).convert("RGB"))
        if depth.shape != rgb.shape[:2]:
            depth = np.array(Image.fromarray(depth, mode="F").resize(
                (rgb.shape[1], rgb.shape[0]), Image.BILINEAR), dtype=np.float32)
        L = luminance(rgb)
        vis = L > args.vignette_threshold
        if vis.sum() < 100:
            continue
        r = pearson(depth[vis], L[vis])
        if not np.isnan(r):
            corrs.append(r)

        if i < args.n_panels:
            far_mask = np.power(np.clip(1.0 - depth, 0, 1), 2.0)  # under-expected
            panel = Image.new("RGB", (rgb.shape[1] * 4, rgb.shape[0]))
            panel.paste(Image.fromarray(rgb), (0, 0))
            panel.paste(gray_to_img(depth), (rgb.shape[1], 0))
            panel.paste(gray_to_img(L / 100.0), (rgb.shape[1] * 2, 0))
            panel.paste(gray_to_img(far_mask), (rgb.shape[1] * 3, 0))
            panel.save(out_dir / f"panel_{s}.png")

    corrs = np.array(corrs, dtype=np.float64)
    mean_r = float(corrs.mean()) if len(corrs) else float("nan")
    print(f"\n[orient] mean Pearson(depth, L) over {len(corrs)} frames = {mean_r:+.3f}")
    print(f"[orient]   min={corrs.min():+.3f}  max={corrs.max():+.3f}  "
          f"frac_positive={float((corrs > 0).mean()):.0%}")
    if mean_r > 0.2:
        print("[verdict] depth is oriented CORRECTLY and informative "
              "(near=bright, far=dark). => the under failure is a TRAINING problem,"
              " not a depth-data problem.")
    elif mean_r < -0.2:
        print("[verdict] depth is INVERTED (near/far swapped). => regenerate depth "
              "or flip the sign; depth will likely help once fixed.")
    else:
        print("[verdict] depth is UNINFORMATIVE (~0 correlation). => HADepth ckpt "
              "probably didn't load; regenerate the depth cache and check the "
              "matched-keys ratio.")
    print(f"\n[panels] wrote {min(args.n_panels, len(stems))} panels to {out_dir}")
    print("         columns: [ RGB | depth(1=near) | luminance | under far-mask ]")


if __name__ == "__main__":
    main()

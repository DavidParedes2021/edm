#!/usr/bin/env python3
"""
prepare_data.py — convert a (normal, exposed) image-pair source into the
folder layout EndoLMSPEC expects, with a leakage-safe train/val/test split.

Output layout (one tree per dataset x exposure-type):

    <out_root>/<name>_<exposure>/
        training/   { exp_images/ , gt_images/ }
        validation/ { exp_images/ , gt_images/ }
        test/       { exp_images/ , gt_images/ }

  * exp_images = the corrupted frame  (over- or under-exposed)
  * gt_images  = the matching normal frame (same stem)
  * exp and gt files are written with the SAME filename (patches_extraction.py
    in EndoLMSPEC reads the exposed file using the gt filename, so they MUST
    match exactly).

The split is keyed on the file STEM via common_split.split_for_stem, so a
frame shared between two datasets always lands in the same split -> no leakage
across the cross-dataset comparison.

Examples
--------
# Your synthetic dataset, over-exposure:
python prepare_data.py --name mine --exposure over \
    --normal_dir   ".../edm_aug_diff_synthetic_dataset/normal_frames" \
    --exposed_dir  ".../edm_aug_diff_synthetic_dataset/overexposed" \
    --out_root     "./compare_data"

# ENDO4IE, over-exposure, restricted to frames shared with the synthetic set
# and capped to the same number of training frames (fair size comparison):
python prepare_data.py --name endo4ie --exposure over \
    --normal_dir   ".../endo4ie/over/normal" \
    --exposed_dir  ".../endo4ie/over/over" \
    --out_root     "./compare_data" \
    --restrict_to  ".../edm_aug_diff_synthetic_dataset/normal_frames" \
    --cap_train    350
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image

from common_split import split_for_stem

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_by_stem(folder: Path) -> dict:
    """Map {stem: path} for all images in a folder (last one wins on clash)."""
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in EXTS:
            out[p.stem] = p
    return out


def save_resized(src: Path, dst: Path, size: int):
    """Load src, resize to size x size, save as dst (PNG)."""
    img = Image.open(src).convert("RGB")
    if size > 0:
        img = img.resize((size, size), Image.BICUBIC)
    img.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset tag, e.g. 'mine' or 'endo4ie'")
    ap.add_argument("--exposure", required=True, choices=["over", "under"])
    ap.add_argument("--normal_dir", required=True, help="dir of clean/normal frames (GT)")
    ap.add_argument("--exposed_dir", required=True, help="dir of corrupted frames (input)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--size", type=int, default=512,
                    help="resize frames to size x size (0 = keep original). EndoLMSPEC assumes 512.")
    ap.add_argument("--restrict_to", default=None,
                    help="optional: dir of frames to intersect stems with "
                         "(use the OTHER dataset's normal_dir to build a common test set)")
    ap.add_argument("--cap_train", type=int, default=0,
                    help="optional: cap the number of TRAINING frames (deterministic, "
                         "by sorted stem) so dataset sizes match. 0 = no cap.")
    ap.add_argument("--salt", default="endolmspec-compare-v1",
                    help="hash salt for the split (keep identical across all datasets!)")
    args = ap.parse_args()

    normal = index_by_stem(Path(args.normal_dir))
    exposed = index_by_stem(Path(args.exposed_dir))

    stems = set(normal) & set(exposed)
    missing = (set(normal) | set(exposed)) - stems
    if missing:
        print(f"[warn] {len(missing)} stems present in only one of normal/exposed "
              f"-> skipped (e.g. {sorted(missing)[:3]})")

    if args.restrict_to:
        keep = set(index_by_stem(Path(args.restrict_to)))
        before = len(stems)
        stems &= keep
        print(f"[restrict] {before} -> {len(stems)} stems after intersecting with "
              f"{args.restrict_to}")

    if not stems:
        raise SystemExit("No usable (normal, exposed) pairs found.")

    # bucket stems by split
    buckets = {"train": [], "validation": [], "test": []}
    for s in sorted(stems):
        buckets[split_for_stem(s, salt=args.salt)].append(s)

    # optional training cap for fair size comparison
    if args.cap_train and len(buckets["train"]) > args.cap_train:
        print(f"[cap] training {len(buckets['train'])} -> {args.cap_train} frames")
        buckets["train"] = buckets["train"][:args.cap_train]

    root = Path(args.out_root) / f"{args.name}_{args.exposure}"
    split_dirname = {"train": "training", "validation": "validation", "test": "test"}

    counts = {}
    for split, slist in buckets.items():
        sd = root / split_dirname[split]
        (sd / "exp_images").mkdir(parents=True, exist_ok=True)
        (sd / "gt_images").mkdir(parents=True, exist_ok=True)
        for s in slist:
            fname = f"{s}.png"  # identical name for exp and gt (required by patches_extraction)
            save_resized(exposed[s], sd / "exp_images" / fname, args.size)
            save_resized(normal[s], sd / "gt_images" / fname, args.size)
        counts[split] = len(slist)

    print(f"[done] {root}")
    print(f"       train={counts['train']}  val={counts['validation']}  test={counts['test']}")


if __name__ == "__main__":
    main()

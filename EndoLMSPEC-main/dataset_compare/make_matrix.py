#!/usr/bin/env python3
"""
make_matrix.py — turn the summary.csv produced by evaluate_pairs.py into a
readable train x test matrix (one per metric) and print the headline verdict.

Usage:
    python make_matrix.py --summary_csv ./results/summary.csv
"""

import argparse
import csv
from collections import defaultdict


def load(summary_csv):
    rows = []
    with open(summary_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def print_matrix(rows, metric, exposure):
    # filter to this exposure (tags look like 'mine_over', 'endo4ie_over')
    sel = [r for r in rows if r["train_tag"].endswith(exposure)
           and r["test_tag"].endswith(exposure)]
    if not sel:
        return
    trains = sorted({r["train_tag"] for r in sel})
    tests = sorted({r["test_tag"] for r in sel})
    cell = defaultdict(dict)
    for r in sel:
        cell[r["train_tag"]][r["test_tag"]] = r[metric]

    w = max([len(t) for t in trains + tests] + [12])
    print(f"\n### {metric.upper()}  —  {exposure}-exposure   (rows = trained on, cols = tested on)")
    print(" " * (w + 2) + "".join(f"{t:>{w+2}}" for t in tests))
    for tr in trains:
        line = f"{tr:<{w+2}}"
        for te in tests:
            line += f"{cell[tr].get(te, '-'):>{w+2}}"
        print(line)


def verdict(rows, exposure):
    """Headline: does training on 'mine' generalize to the other dataset better
    than training on 'endo4ie' generalizes to mine?"""
    def get(tr, te, metric="ssim_mean"):
        for r in rows:
            if r["train_tag"] == tr and r["test_tag"] == te:
                return float(r[metric])
        return None

    mine, endo = f"mine_{exposure}", f"endo4ie_{exposure}"
    cross_mine = get(mine, endo)     # train mine, test on endo4ie
    cross_endo = get(endo, mine)     # train endo4ie, test on mine
    diag_mine = get(mine, mine)
    diag_endo = get(endo, endo)

    print(f"\n--- VERDICT ({exposure}-exposure, SSIM) ---")
    if None in (cross_mine, cross_endo):
        print("  (need both cross cells filled — run the full matrix first)")
        return
    print(f"  in-distribution:   mine->mine={diag_mine}   endo4ie->endo4ie={diag_endo}")
    print(f"  generalization:    mine->endo4ie={cross_mine:.4f}   endo4ie->mine={cross_endo:.4f}")
    if cross_mine > cross_endo:
        print(f"  [absolute] training on YOUR data generalizes better "
              f"(+{cross_mine - cross_endo:.4f} SSIM).")
    elif cross_endo > cross_mine:
        print(f"  [absolute] training on ENDO4IE generalizes better "
              f"(+{cross_endo - cross_mine:.4f} SSIM).")
    else:
        print("  [absolute] tie.")

    # The two test sets are different scenes (EAD/EDD vs Kvasir), so the absolute
    # cross numbers are partly a content effect. The drop from each model's own
    # diagonal to its off-diagonal is a content-normalised robustness measure:
    # how much does a model degrade when it leaves its training distribution?
    if None not in (diag_mine, diag_endo):
        drop_mine = diag_mine - cross_mine     # mine: in-dist -> out-of-dist
        drop_endo = diag_endo - cross_endo
        print(f"  [robustness] generalisation drop  mine={drop_mine:+.4f}  endo4ie={drop_endo:+.4f}"
              f"  (smaller drop = more transferable training data)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    args = ap.parse_args()
    rows = load(args.summary_csv)
    for exposure in ("over", "under"):
        for metric in ("ssim_mean", "psnr_mean", "lpips_mean"):
            print_matrix(rows, metric, exposure)
        verdict(rows, exposure)


if __name__ == "__main__":
    main()

"""
common_split.py — deterministic, dataset-independent train/val/test split.

Both the synthetic dataset and ENDO4IE are derived from overlapping source
frames (e.g. EAD2020_frameOnly_*). If we split each dataset independently we
risk a frame being TRAIN for one model and TEST for another -> leakage that
invalidates the cross-dataset comparison.

The fix: assign each frame to a split using a hash of its *stem* only. The
assignment is therefore identical across every dataset, so any shared frame
always lands in the same split. No frame is ever train-here / test-there.

Default split: 70% train / 15% val / 15% test.
"""

import hashlib


def split_for_stem(stem: str, salt: str = "endolmspec-compare-v1",
                   train_frac: float = 0.70, val_frac: float = 0.15) -> str:
    """Return 'train' | 'validation' | 'test' for a given file stem.

    Deterministic: same stem -> same split, regardless of dataset or run.
    """
    h = hashlib.md5(f"{salt}:{stem}".encode("utf-8")).hexdigest()
    # map first 8 hex chars to a float in [0, 1)
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "validation"
    return "test"

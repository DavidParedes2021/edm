#!/usr/bin/env python
"""
scripts/create_dummy_data.py
Creates a minimal dummy dataset for smoke-testing the pipeline
on a laptop without real data.

Generates:
  - 20 train pairs, 5 val pairs, 5 test pairs
  - For both under and over-exposed
  - Images are random RGB noise (just enough to verify shapes/devices)

Usage:
  python scripts/create_dummy_data.py
  python scripts/train.py --config configs/debug.yaml
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path


SPLITS = {
    "train":      20,
    "validation":  5,
    "test":        5,
}

ROOTS = {
    "underexposed": {
        "artifact_key": "underexposed",
        "normal_key":   "normal_frames",
        "artifact_fn":  lambda img: np.clip(img * 0.3 + np.random.randint(-10, 0), 0, 255).astype(np.uint8),
    },
    "overexposed": {
        "artifact_key": "overexposed",
        "normal_key":   "normal_frames",
        "artifact_fn":  lambda img: np.clip(img * 1.7 + np.random.randint(20, 60), 0, 255).astype(np.uint8),
    },
}

IMAGE_SIZE = 64  # tiny for speed


def make_normal_image(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Smooth gradient + noise to vaguely resemble a real image
    base = rng.randint(60, 200, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return base


def create_dummy_dataset():
    print("Creating dummy dataset...")

    for root_name, info in ROOTS.items():
        root = Path(f"real_{root_name}")
        for split, n in SPLITS.items():
            norm_dir = root / split / info["normal_key"]
            art_dir  = root / split / info["artifact_key"]
            norm_dir.mkdir(parents=True, exist_ok=True)
            art_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                filename = f"frame_{i:04d}.png"
                normal_np = make_normal_image(seed=i + hash(root_name + split) % 1000)
                artifact_np = info["artifact_fn"](normal_np.copy())

                Image.fromarray(normal_np).save(str(norm_dir / filename))
                Image.fromarray(artifact_np).save(str(art_dir / filename))

            print(f"  {root}/{split}/: {n} pairs")

    print("\nDummy dataset created successfully.")
    print("\nYou can now run:")
    print("  python scripts/train.py --config configs/debug.yaml")


if __name__ == "__main__":
    create_dummy_dataset()

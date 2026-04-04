"""
generate_dummy_data.py — Create tiny synthetic datasets for smoke testing.

Run this BEFORE train.py to populate data/normal, data/overexposed and
data/underexposed with dummy PNG images so the pipeline can be verified
end-to-end on any machine without real data.

Usage:
    python generate_dummy_data.py          # default: 20 images per domain
    python generate_dummy_data.py --n 5    # minimal test
"""

import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image


def make_normal(size=256) -> Image.Image:
    """Synthetic 'normal' frame: random mid-tones."""
    rng   = np.random.RandomState()
    base  = rng.randint(80, 180, (size, size, 3), dtype=np.uint8)
    noise = rng.randint(0, 30, (size, size, 3), dtype=np.uint8)
    arr   = np.clip(base.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    # Add a simple gradient to simulate a scene
    for r in range(size):
        arr[r, :, :] = np.clip(arr[r, :, :].astype(int) + r // 4, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_overexposed(img: Image.Image, ev: float = 2.0) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr * (2.0 ** ev), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_underexposed(img: Image.Image, ev: float = -2.0) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * (2.0 ** ev)
    sigma = 0.02 * (2.0 ** (-ev))
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=20, help="Images per domain")
    p.add_argument("--size", type=int, default=256, help="Image size in pixels")
    args = p.parse_args()

    dirs = {
        "normal":      Path("data/normal"),
        "overexposed": Path("data/overexposed"),
        "underexposed": Path("data/underexposed"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    for i in range(args.n):
        normal_img = make_normal(args.size)
        normal_img.save(dirs["normal"] / f"frame_{i:04d}.png")

        # Overexposed: generate from a DIFFERENT Normal frame (unpaired)
        over_base = make_normal(args.size)
        over_img  = make_overexposed(over_base, ev=random.uniform(1.5, 3.0))
        over_img.save(dirs["overexposed"] / f"frame_{i:04d}.png")

        # Underexposed: same — from a different base
        under_base = make_normal(args.size)
        under_img  = make_underexposed(under_base, ev=random.uniform(-3.0, -1.5))
        under_img.save(dirs["underexposed"] / f"frame_{i:04d}.png")

    print(f"Created {args.n} synthetic images per domain in data/")
    for k, d in dirs.items():
        imgs = list(d.glob("*.png"))
        print(f"  {k:15s}: {len(imgs)} images in {d}")


if __name__ == "__main__":
    main()

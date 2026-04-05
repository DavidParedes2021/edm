#!/usr/bin/env python
"""
scripts/generate.py
Generate a synthetic paired dataset using a trained model.

Usage:
  python scripts/generate.py \\
      --config configs/train.yaml \\
      --checkpoint outputs/train/checkpoints/best.pt \\
      --input_dir path/to/normal_frames/ \\
      --output_dir path/to/synthetic_output/ \\
      --exposure over   \\  # "over", "under", or "both"
      --strength 0.8    \\  # 0.0-1.0 (how strong the effect)
      --steps 50

Output structure:
  output_dir/
    overexposed/   ← generated artifacts
    underexposed/  ← generated artifacts
    normal/        ← copied normal frames (ground truth)
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models.unet     import build_model
from models.diffusion import build_scheduler, build_sampler, DDIMSampler
from training.config_utils import load_config, set_seed


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def collect_images(directory: str):
    p = Path(directory)
    return sorted([f for f in p.iterdir() if f.suffix.lower() in VALID_EXT])


def load_normal(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, (image_size, image_size),
                    interpolation=TF.InterpolationMode.BICUBIC)
    t   = T.ToTensor()(img)                  # [3,H,W] in [0,1]
    t   = T.Normalize([0.5]*3, [0.5]*3)(t)  # → [-1,1]
    return t.unsqueeze(0).to(device)         # [1,3,H,W]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.float().squeeze(0).clamp(-1, 1) + 1.0) * 0.5 * 255).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def load_checkpoint_weights(checkpoint_path: str, model, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and Trainer checkpoint format
    if "model" in state:
        model.load_state_dict(state["model"])
        # Prefer EMA weights if available
        if "ema" in state:
            ema_shadow = state["ema"]["shadow"]
            sd = model.state_dict()
            for k in sd:
                if k in ema_shadow:
                    sd[k] = ema_shadow[k].to(device)
            model.load_state_dict(sd)
            print("  Using EMA weights.")
    else:
        model.load_state_dict(state)
    print(f"  Loaded checkpoint: {checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    sampler: DDIMSampler,
    normal_paths: list,
    output_subdir: str,
    exposure_value: float,
    image_size: int,
    device: torch.device,
    batch_size: int = 1,
):
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    model.eval()

    for i in tqdm(range(0, len(normal_paths), batch_size), desc=f"Generating (exp={exposure_value:+.1f})"):
        batch_paths = normal_paths[i : i + batch_size]
        normals = []
        for p in batch_paths:
            normals.append(load_normal(str(p), image_size, device))
        cond     = torch.cat(normals, dim=0)                    # [B,3,H,W]
        exposure = torch.full((len(batch_paths),), exposure_value, device=device)

        B, C, H, W = cond.shape
        generated  = sampler.sample(
            model    = model,
            shape    = (B, C, H, W),
            cond     = cond,
            exposure = exposure,
            device   = device,
        )

        for j, src_path in enumerate(batch_paths):
            out_img  = tensor_to_pil(generated[j:j+1])
            out_name = Path(src_path).stem + ".png"
            out_path = os.path.join(output_subdir, out_name)
            out_img.save(out_path)

        # Free GPU memory
        del generated, cond, exposure, normals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic illumination artifacts")
    p.add_argument("--config",     required=True,  help="Path to YAML config")
    p.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    p.add_argument("--input_dir",  required=True,  help="Directory of normal frames")
    p.add_argument("--output_dir", required=True,  help="Output directory")
    p.add_argument("--exposure",   default="both",
                   choices=["over", "under", "both"],
                   help="Which exposure type to generate")
    p.add_argument("--strength",   type=float, default=1.0,
                   help="Exposure strength in [0,1]")
    p.add_argument("--steps",      type=int,   default=50,
                   help="DDIM inference steps")
    p.add_argument("--batch_size", type=int,   default=1,
                   help="Batch size for generation")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Input dir  : {args.input_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Exposure   : {args.exposure} (strength={args.strength})")
    print(f"DDIM steps : {args.steps}")

    # Override inference steps from CLI
    cfg["inference"]["num_inference_steps"] = args.steps

    # Build model
    model     = build_model(cfg).to(device)
    scheduler = build_scheduler(cfg)
    sampler   = build_sampler(cfg, scheduler)

    load_checkpoint_weights(args.checkpoint, model, device)

    # Collect input images
    normal_paths = collect_images(args.input_dir)
    if len(normal_paths) == 0:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(normal_paths)} normal frames.")

    out_root = Path(args.output_dir)

    # Copy normal frames as ground truth
    normal_out = out_root / "normal"
    normal_out.mkdir(parents=True, exist_ok=True)
    for p in tqdm(normal_paths, desc="Copying normal frames"):
        shutil.copy(str(p), str(normal_out / (p.stem + ".png")))

    # Generate
    if args.exposure in ("over", "both"):
        generate(
            model      = model,
            sampler    = sampler,
            normal_paths= normal_paths,
            output_subdir= str(out_root / "overexposed"),
            exposure_value= float(args.strength),
            image_size = cfg["data"]["image_size"],
            device     = device,
            batch_size = args.batch_size,
        )

    if args.exposure in ("under", "both"):
        generate(
            model      = model,
            sampler    = sampler,
            normal_paths= normal_paths,
            output_subdir= str(out_root / "underexposed"),
            exposure_value= float(-args.strength),
            image_size = cfg["data"]["image_size"],
            device     = device,
            batch_size = args.batch_size,
        )

    print(f"\nDone. Synthetic dataset written to: {args.output_dir}")
    print("Structure:")
    print(f"  {args.output_dir}/normal/          ← ground truth normal frames")
    if args.exposure in ("over", "both"):
        print(f"  {args.output_dir}/overexposed/    ← generated over-exposed")
    if args.exposure in ("under", "both"):
        print(f"  {args.output_dir}/underexposed/   ← generated under-exposed")


if __name__ == "__main__":
    main()

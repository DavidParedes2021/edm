#!/usr/bin/env python3
"""
inference.py — Generate overexposed/underexposed versions of normal frames.

Usage:
    python inference.py --config config.yaml --checkpoint output/checkpoints/best.pt
    python inference.py --config config.yaml --checkpoint output/checkpoints/best.pt \
        --domain overexposed --noise_strength 0.5 --guidance_scale 12
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm
from PIL import Image

import yaml

from dataset import NormalImageDataset, lab_to_rgb_numpy
from train import build_model, EMAModel, load_config


@torch.no_grad()
def run_inference(cfg: dict, checkpoint_path: str, overrides: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] device = {device}")

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(cfg, device)

    # use EMA weights
    if "ema" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(model)
        print("[Inference] loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])

    model.eval()

    # override inference params
    noise_strength = overrides.get("noise_strength", cfg["inference"]["noise_strength"])
    guidance_scale = overrides.get("guidance_scale", cfg["inference"]["guidance_scale"])
    num_steps = overrides.get("num_inference_steps", cfg["inference"]["num_inference_steps"])
    domain = overrides.get("domain", "both")  # "overexposed", "underexposed", "both"

    # setup scheduler
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    ddim.set_timesteps(num_steps, device=device)

    total_steps = len(ddim.timesteps)
    start_step = int(total_steps * (1 - noise_strength))
    start_step = max(0, min(start_step, total_steps - 1))

    # dataset
    normal_ds = NormalImageDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
    )
    loader = DataLoader(
        normal_ds,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # output dir
    out_root = Path(cfg["output"]["root"]) / "generated"
    out_root.mkdir(parents=True, exist_ok=True)

    domains = []
    if domain in ("overexposed", "both"):
        domains.append((0, "overexposed"))
    if domain in ("underexposed", "both"):
        domains.append((1, "underexposed"))

    for domain_label, domain_name in domains:
        out_dir = out_root / domain_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Generating] {domain_name} (guidance={guidance_scale}, "
              f"noise_strength={noise_strength}, steps={num_steps})")

        for L_tensor, AB_tensor, paths in tqdm(loader, desc=domain_name):
            L_tensor = L_tensor.to(device)
            AB_np = AB_tensor.numpy()
            B = L_tensor.shape[0]

            noise = torch.randn_like(L_tensor)
            t_start = ddim.timesteps[start_step]
            x = ddim.add_noise(L_tensor, noise, torch.tensor([t_start], device=device))

            cond = torch.full((B,), domain_label, dtype=torch.long, device=device)
            uncond = torch.full((B,), 2, dtype=torch.long, device=device)

            for t in ddim.timesteps[start_step:]:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)
                with autocast(enabled=cfg["training"]["mixed_precision"]):
                    p_c = model(x, t_batch, class_labels=cond).sample
                    p_u = model(x, t_batch, class_labels=uncond).sample
                pred = p_u + guidance_scale * (p_c - p_u)
                x = ddim.step(pred, t, x).prev_sample

            L_out = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                lab = np.stack([L_out[b], AB_np[b, :, :, 0], AB_np[b, :, :, 1]], axis=-1)
                rgb = lab_to_rgb_numpy(lab)
                stem = Path(paths[b]).stem
                Image.fromarray(rgb).save(str(out_dir / f"{stem}.png"))

    print(f"\n[Done] outputs in {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="both",
                        choices=["overexposed", "underexposed", "both"])
    parser.add_argument("--noise_strength", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {"domain": args.domain}
    if args.noise_strength is not None:
        overrides["noise_strength"] = args.noise_strength
    if args.guidance_scale is not None:
        overrides["guidance_scale"] = args.guidance_scale
    if args.num_inference_steps is not None:
        overrides["num_inference_steps"] = args.num_inference_steps

    run_inference(cfg, args.checkpoint, overrides)


if __name__ == "__main__":
    main()

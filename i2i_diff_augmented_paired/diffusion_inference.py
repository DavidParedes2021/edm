#!/usr/bin/env python3
"""
diffusion_inference.py — Generate over/underexposed images from the trained model.

The key to texture preservation at inference:
    1. Diffusion predicts a target L at model resolution (256×256)
    2. Upsample the predicted L to original resolution
    3. Decompose: L_predicted = low-freq envelope + (discard predicted texture)
    4. Decompose: L_original  = (discard envelope) + high-freq texture
    5. Recombine: L_final = predicted envelope + original texture
    6. Merge with untouched A,B → RGB

This is the same texture-injection trick from exposure_augment.py.
The diffusion provides the *exposure envelope*; the original provides the *texture*.

Usage:
    python diffusion_inference.py \
        --config diffusion_config.yaml \
        --checkpoint ./output_diffusion/checkpoints/best.pt

    # Overexposed only, custom output:
    python diffusion_inference.py \
        --config diffusion_config.yaml \
        --checkpoint ./output_diffusion/checkpoints/best.pt \
        --domain overexposed \
        --output_dir ./my_output
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image
import yaml

from diffusion_dataset import NormalInferenceDataset
from diffusion_train import build_model, EMAModel, load_config
from exposure_augment import lab_to_rgb


@torch.no_grad()
def run_inference(
    cfg: dict,
    checkpoint_path: str,
    domain: str = "both",
    output_dir: str = None,
    texture_sigma_base: float = 3.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] device = {device}")

    # load model with EMA weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(cfg, device)
    if "ema" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(model)
        print("[Inference] loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # scheduler
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    num_steps = cfg["inference"]["num_inference_steps"]
    ddim.set_timesteps(num_steps, device=device)

    # dataset
    normal_ds = NormalInferenceDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
    )
    loader = DataLoader(
        normal_ds,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # output
    out_root = Path(output_dir) if output_dir else Path(cfg["output"]["root"]) / "generated"

    domains = []
    if domain in ("overexposed", "both"):
        domains.append((0, "overexposed"))
    if domain in ("underexposed", "both"):
        domains.append((1, "underexposed"))

    for domain_label, domain_name in domains:
        out_dir = out_root / domain_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Generating] {domain_name} ({num_steps} DDIM steps)")

        for source_L, AB, L_orig_tensor, paths, orig_hw in tqdm(loader, desc=domain_name):
            source_L = source_L.to(device)  # (B, 1, H, W)
            B = source_L.shape[0]

            # start from pure noise for the target channel
            x = torch.randn_like(source_L)
            cond = torch.full((B,), domain_label, dtype=torch.long, device=device)

            # full denoising loop
            for t in ddim.timesteps:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)
                model_input = torch.cat([x, source_L], dim=1)  # (B, 2, H, W)
                with autocast(enabled=cfg["training"]["mixed_precision"]):
                    pred_noise = model(model_input, t_batch, class_labels=cond).sample
                x = ddim.step(pred_noise, t, x).prev_sample

            # denormalise predicted L: [-1, 1] → [0, 100]
            L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                H_orig, W_orig = int(orig_hw[0][b]), int(orig_hw[1][b])
                ab = AB[b].numpy()           # (H_orig, W_orig, 2)
                L_orig_np = L_orig_tensor[b].numpy()  # (H_orig, W_orig)

                # upsample predicted L to original resolution
                L_pred_full = np.array(
                    Image.fromarray(L_pred_small[b].astype(np.float32), mode="F").resize(
                        (W_orig, H_orig), Image.LANCZOS
                    ),
                    dtype=np.float32,
                )

                # ── Texture preservation via frequency decomposition ──
                sigma = texture_sigma_base * max(H_orig, W_orig) / 512.0

                # original: keep only high-pass (texture)
                L_orig_low = gaussian_filter(L_orig_np, sigma=sigma)
                L_high = L_orig_np - L_orig_low

                # predicted: keep only low-pass (exposure envelope)
                L_pred_low = gaussian_filter(L_pred_full, sigma=sigma)

                # recombine: predicted envelope + original texture
                L_final = np.clip(L_pred_low + L_high, 0.0, 100.0).astype(np.float32)

                # reconstruct RGB
                lab = np.stack([L_final, ab[..., 0], ab[..., 1]], axis=-1)
                rgb = lab_to_rgb(lab)

                stem = Path(paths[b]).stem
                Image.fromarray(rgb).save(str(out_dir / f"{stem}.png"))

    print(f"\n[Done] outputs in {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="both",
                        choices=["overexposed", "underexposed", "both"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--texture_sigma", type=float, default=3.0,
                        help="Gaussian sigma for texture decomposition (at 512px ref)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(
        cfg, args.checkpoint,
        domain=args.domain,
        output_dir=args.output_dir,
        texture_sigma_base=args.texture_sigma,
    )


if __name__ == "__main__":
    main()

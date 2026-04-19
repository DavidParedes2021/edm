#!/usr/bin/env python3
"""
diffusion_inference.py — Depth-aware inference for exposure-augmented frames.

Pipeline at each frame:
    1. Load RGB → LAB; keep AB untouched.
    2. Load / compute depth map (from --depth_dir cache, or on-the-fly with
       Depth Anything V2 via NormalInferenceDataset).
    3. Downsample (L, depth) to model resolution (256×256).
    4. DDIM denoise. Model input is 3-channel: [noisy_target_L, source_L, depth].
       Class embedding selects overexposed (0) or underexposed (1).
    5. Upsample predicted L to original resolution.
    6. Texture reinjection via frequency decomposition:
         L_final = low_pass(L_pred) + high_pass(L_orig)
    7. Depth-aware chroma attenuation on (A, B) conditioned on (L_new, depth).
       This is the fix for the "brownish underexposure" failure mode — low L
       with unchanged chroma reads as brown; we desaturate toward neutral as
       L → 0, with extra desat for far regions.
    8. LAB → RGB, save.

Usage:
    python diffusion_inference.py \\
        --config diffusion_config.yaml \\
        --checkpoint ./outputs/checkpoints/best.pt

    # Underexposed only, with pre-computed depth cache:
    python diffusion_inference.py \\
        --config diffusion_config.yaml \\
        --checkpoint ./outputs/checkpoints/best.pt \\
        --domain underexposed \\
        --depth_dir ./data/depth \\
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

from diffusion_dataset import NormalInferenceDataset
from diffusion_train import build_model, EMAModel, load_config
from exposure_augment import lab_to_rgb
from chroma_attenuation import attenuate_chroma, DEFAULT_CFG as CHROMA_DEFAULT_CFG


@torch.no_grad()
def run_inference(
    cfg: dict,
    checkpoint_path: str,
    domain: str = "both",
    output_dir: str = None,
    depth_dir: str = None,
    texture_sigma_base: float = 3.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] device = {device}")

    # ── model ────────────────────────────────────────────────────────────
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

    # ── scheduler ────────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    num_steps = cfg["inference"]["num_inference_steps"]
    ddim.set_timesteps(num_steps, device=device)

    # ── dataset ──────────────────────────────────────────────────────────
    # prefer CLI arg, else config (data.depth_dir), else on-the-fly
    resolved_depth_dir = depth_dir or cfg.get("data", {}).get("depth_dir")
    if resolved_depth_dir and not Path(resolved_depth_dir).is_dir():
        print(f"[Inference] depth_dir '{resolved_depth_dir}' not found; "
              f"depth will be computed on-the-fly.")
        resolved_depth_dir = None

    normal_ds = NormalInferenceDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
        depth_dir=resolved_depth_dir,
    )
    loader = DataLoader(
        normal_ds,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── chroma cfg ───────────────────────────────────────────────────────
    chroma_cfg = dict(CHROMA_DEFAULT_CFG)
    chroma_cfg.update(cfg.get("chroma", {}))

    # ── output root ──────────────────────────────────────────────────────
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

        for source_L, depth, AB, L_orig_tensor, depth_full_tensor, paths, orig_hw in tqdm(
            loader, desc=domain_name
        ):
            source_L = source_L.to(device)  # (B, 1, h, w) in [-1, 1]
            depth    = depth.to(device)     # (B, 1, h, w) in [-1, 1]
            B = source_L.shape[0]

            # start from pure noise for the target L channel
            x = torch.randn_like(source_L)
            cond = torch.full((B,), domain_label, dtype=torch.long, device=device)

            # DDIM loop
            for t in ddim.timesteps:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)
                model_input = torch.cat([x, source_L, depth], dim=1)  # (B, 3, h, w)
                with autocast(enabled=cfg["training"]["mixed_precision"]):
                    pred_noise = model(model_input, t_batch, class_labels=cond).sample
                x = ddim.step(pred_noise, t, x).prev_sample

            # denormalise predicted L: [-1, 1] → [0, 100]
            L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                H_o, W_o = int(orig_hw[0][b]), int(orig_hw[1][b])
                ab    = AB[b].numpy()              # (H_o, W_o, 2), float32
                l_o   = L_orig_tensor[b].numpy()   # (H_o, W_o)
                d_o   = depth_full_tensor[b].numpy()  # (H_o, W_o), [0, 1]

                # upsample predicted L to original resolution
                L_pred_full = np.array(
                    Image.fromarray(L_pred_small[b].astype(np.float32), mode="F").resize(
                        (W_o, H_o), Image.LANCZOS
                    ),
                    dtype=np.float32,
                )

                # ── Texture reinjection via frequency decomposition ──────
                sigma = texture_sigma_base * max(H_o, W_o) / 512.0
                L_high     = l_o - gaussian_filter(l_o, sigma=sigma)         # orig HF
                L_low_pred = gaussian_filter(L_pred_full, sigma=sigma)       # pred LF
                L_final = np.clip(L_low_pred + L_high, 0.0, 100.0).astype(np.float32)

                # ── Depth-aware chroma attenuation ───────────────────────
                # This is the key fix for "brownish underexposure". As L drops
                # toward 0 the chroma multiplier rolls to `sat_floor`; far
                # regions (low depth) get an additional desat; highlights
                # get a separate desat. Purkinje B-shift nudges very dark
                # pixels toward blue instead of green/brown.
                A_new, B_new = attenuate_chroma(
                    ab[..., 0], ab[..., 1],
                    L_new=L_final, L_orig=l_o, depth=d_o,
                    mode=domain_name, cfg=chroma_cfg,
                )

                lab = np.stack([L_final, A_new, B_new], axis=-1)
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
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Directory of pre-computed depth .npy files "
                             "(one per normal frame). If omitted, depth is "
                             "computed on-the-fly via Depth Anything V2.")
    parser.add_argument("--texture_sigma", type=float, default=3.0,
                        help="Gaussian sigma for texture decomposition (at 512px ref).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(
        cfg, args.checkpoint,
        domain=args.domain,
        output_dir=args.output_dir,
        depth_dir=args.depth_dir,
        texture_sigma_base=args.texture_sigma,
    )


if __name__ == "__main__":
    main()

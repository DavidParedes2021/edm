#!/usr/bin/env python3
"""
diffusion_inference.py — Single-domain depth-aware inference.

Each checkpoint is dedicated to one direction (overexposed OR underexposed).
Run inference twice with different --checkpoint and --domain to cover both.

Pipeline at each frame:
    1. Load RGB → LAB; keep AB untouched.
    2. Load / compute depth map (from --depth_dir cache, or on-the-fly).
    3. Downsample (L, depth) to model resolution (256×256).
    4. DDIM denoise. Model input is 3-channel: [noisy_target_L, source_L, depth].
       No class embedding — the model is fully specialised to its domain.
    5. Upsample predicted L to original resolution.
    6. Texture reinjection with HF gate — original high-pass is attenuated
       toward `texture_gate.floor` in extreme cores so dark regions stay dark
       and bright regions stay bright (this is what the previous build was
       missing, causing "barely noticeable" effects).
         L_final = low_pass(L_pred) + hf_gate(L_pred, mode) * high_pass(L_orig)
    7. Depth-aware chroma attenuation on (A, B).
    8. LAB → RGB, save.

Usage:
    python diffusion_inference.py \\
        --config diffusion_config.yaml \\
        --checkpoint .../checkpoints/under/best.pt \\
        --domain underexposed \\
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
from chroma_attenuation import attenuate_chroma, texture_gate, DEFAULT_CFG as CHROMA_DEFAULT_CFG


@torch.no_grad()
def run_inference(
    cfg: dict,
    checkpoint_path: str,
    domain: str = None,
    output_dir: str = None,
    depth_dir: str = None,
    texture_sigma_base: float = 3.0,
):
    # Resolve domain: CLI arg overrides config.
    domain = (domain or cfg.get("domain") or "").strip().lower()
    if domain not in ("overexposed", "underexposed"):
        raise ValueError(
            f"`domain` must be 'overexposed' or 'underexposed' "
            f"(got {domain!r}). Set in config or pass --domain."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] device = {device}, domain = {domain}")

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

    # ── texture-gate cfg ─────────────────────────────────────────────────
    tg_cfg = cfg.get("texture_gate", {}) or {}
    tg_knee = float(tg_cfg.get("knee", 30.0))
    tg_floor = float(tg_cfg.get("floor", 0.05))

    # ── output root ──────────────────────────────────────────────────────
    out_root = (Path(output_dir) if output_dir
                else Path(cfg["output"]["root"]) / "generated")
    out_dir = out_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Generating] {domain} ({num_steps} DDIM steps) → {out_dir}")

    for source_L, depth, AB, L_orig_tensor, depth_full_tensor, paths, orig_hw in tqdm(
        loader, desc=domain
    ):
        source_L = source_L.to(device)  # (B, 1, h, w) in [-1, 1]
        depth    = depth.to(device)     # (B, 1, h, w) in [-1, 1]
        B = source_L.shape[0]

        # start from pure noise for the target L channel
        x = torch.randn_like(source_L)

        # DDIM loop — no class labels, model is single-domain.
        for t in ddim.timesteps:
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            model_input = torch.cat([x, source_L, depth], dim=1)  # (B, 3, h, w)
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred_noise = model(model_input, t_batch,
                                   class_labels=None).sample
            x = ddim.step(pred_noise, t, x).prev_sample

        # denormalise predicted L: [-1, 1] → [0, 100]
        L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

        for b in range(B):
            H_o, W_o = int(orig_hw[0][b]), int(orig_hw[1][b])
            ab    = AB[b].numpy()                  # (H_o, W_o, 2)
            l_o   = L_orig_tensor[b].numpy()       # (H_o, W_o)
            d_o   = depth_full_tensor[b].numpy()   # (H_o, W_o), [0, 1]

            # upsample predicted L to original resolution
            L_pred_full = np.array(
                Image.fromarray(L_pred_small[b].astype(np.float32),
                                mode="F").resize(
                    (W_o, H_o), Image.LANCZOS
                ),
                dtype=np.float32,
            )

            # ── Texture reinjection with HF gate ─────────────────────
            # Without the gate, the original high-pass re-inserts mid-tone
            # variance into pixels the model just drove to extremes,
            # neutralising the effect.
            sigma = texture_sigma_base * max(H_o, W_o) / 512.0
            L_high     = l_o - gaussian_filter(l_o, sigma=sigma)
            L_low_pred = gaussian_filter(L_pred_full, sigma=sigma)
            hf_gate = texture_gate(L_low_pred, mode=domain,
                                   knee=tg_knee, floor=tg_floor)
            L_final = np.clip(L_low_pred + hf_gate * L_high,
                              0.0, 100.0).astype(np.float32)

            # ── Depth-aware chroma attenuation ───────────────────────
            A_new, B_new = attenuate_chroma(
                ab[..., 0], ab[..., 1],
                L_new=L_final, L_orig=l_o, depth=d_o,
                mode=domain, cfg=chroma_cfg,
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
    parser.add_argument("--domain", type=str, default=None,
                        choices=["overexposed", "underexposed"],
                        help="Which direction this checkpoint targets. "
                             "Defaults to the 'domain' field in the config.")
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

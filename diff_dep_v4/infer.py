"""YCLDI inference: SDEdit + RePaint with depth-aware focalization masks.

Usage
-----
    python infer.py --config config.yaml --input <file_or_dir>
    python infer.py --config config.yaml --input frames/ --output gen/ --ckpt best

For each normal frame this script writes three images to the output directory:
    <stem>_normal.png         (preprocessed original, for the paired dataset)
    <stem>_overexposed.png    (focalized synthetic overexposure)
    <stem>_underexposed.png   (focalized synthetic underexposure)

If ``inference.save_debug`` is true in the config it also dumps the pseudo-depth
map and the over/under vulnerability masks under ``<output>/debug/``.

Algorithm (per frame, per target class)
---------------------------------------
1. Load RGB -> square center-crop -> resize -> tensor.
2. Split YCbCr.  Diffusion only touches Y; Cb/Cr go straight to the output.
3. Compute pseudo-depth from RGB luminance (dark = far, bright = close).
4. Compute focalized vulnerability mask:
       mask_over   prop. to  Y * (1 - depth)         (bright + close)
       mask_under  prop. to  (1 - Y) * depth         (dark   + far  )
5. SDEdit-noise Y to t = strength * T.
6. DDIM denoise with classifier-free guidance, contrastive against the
   ``normal`` class by default so the network is pushed AWAY from normal
   AND TOWARD the target exposure.  At every step, RePaint-blend with the
   noised original outside the mask -> the change stays focal.
7. Final blend  y_final = mask * y_gen + (1 - mask) * y_in  -> merges the
   generated cluster back into the original frame at full resolution.
8. Recombine Y_final with the original Cb/Cr  -> RGB output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from diffusers import DDIMScheduler

from colorspace import merge_ycbcr, split_ycbcr
from dataset import (
    CLASS_NORMAL,
    CLASS_NULL,
    CLASS_OVER,
    CLASS_UNDER,
    IMG_EXTS,
)
from ema import EMA
from mask import depth_from_rgb, vulnerability_masks
from model import build_unet
from utils import derive_output_paths, load_config, set_seed


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Path, image_size: int) -> torch.Tensor:
    """Load RGB -> (1, 3, H, W) float in [0, 1]; square center-crop + bicubic resize.

    Mirrors ``YCbCrEndoscopyDataset._load_rgb`` so train/inference preprocessing
    stay byte-identical.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((image_size, image_size), Image.BICUBIC)
    return TF.to_tensor(img).unsqueeze(0)


def save_rgb(rgb: torch.Tensor, path: Path) -> None:
    """rgb: (1, 3, H, W) in [0, 1] -> PNG."""
    arr = (rgb.detach().clamp(0.0, 1.0).cpu().float().numpy()[0]
           .transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def save_gray(t: torch.Tensor, path: Path) -> None:
    """t: (1, 1, H, W) in [0, 1] -> 8-bit grayscale PNG."""
    arr = (t.detach().clamp(0.0, 1.0).cpu().float().numpy()[0, 0] * 255
           ).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))


def list_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = [p for p in input_path.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
        return sorted(files)
    raise FileNotFoundError(f"Input not found: {input_path}")


# ---------------------------------------------------------------------------
# Diffusion: SDEdit + classifier-free guidance + step-wise RePaint
# ---------------------------------------------------------------------------

@torch.no_grad()
def sdedit_with_mask(
    unet,
    scheduler: DDIMScheduler,
    y_in: torch.Tensor,            # (B, 1, H, W) in [-1, 1]  -- input Y channel
    mask: torch.Tensor,            # (B, 1, H, W) in  [0, 1]   -- focal mask
    target_class: int,
    negative_class: int,
    cfg_scale: float,
    strength: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate the target-class luminance with focalization + CFG.

    Returns y_gen in [-1, 1] of shape (B, 1, H, W).  The caller is responsible
    for the final mask blend; this function applies *step-wise* RePaint
    blending so the unmasked region tracks the noised original at every
    timestep, which both keeps the network focused on the focal cluster and
    makes the blend boundary visually consistent.
    """
    B = y_in.shape[0]

    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)            # high -> low
    init_step_idx = int(round((1.0 - strength) * num_steps))
    init_step_idx = max(0, min(init_step_idx, num_steps - 1))
    sub_timesteps = timesteps[init_step_idx:]

    # SDEdit: noise the input to the highest timestep we will denoise from.
    t_start = sub_timesteps[0]
    noise = torch.randn_like(y_in)
    t_start_batched = torch.full((B,), int(t_start.item()), device=device, dtype=torch.long)
    x = scheduler.add_noise(y_in, noise, t_start_batched)

    pos_labels = torch.full((B,), target_class,   device=device, dtype=torch.long)
    neg_labels = torch.full((B,), negative_class, device=device, dtype=torch.long)

    for i, t in enumerate(sub_timesteps):
        t_batched = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

        # Two forward passes for CFG.  Concatenating into a single batch is a
        # bit faster but doubles activation memory; on a 16 GB DGX with a
        # single 256x256 sample two sequential passes are cheap enough and
        # keep peak memory low so larger inputs / batches stay safe.
        eps_pos = unet(x, t_batched, class_labels=pos_labels).sample
        eps_neg = unet(x, t_batched, class_labels=neg_labels).sample
        eps_cfg = eps_neg + cfg_scale * (eps_pos - eps_neg)

        # DDIM step -> prev_sample at the *next* timestep
        x_denoised = scheduler.step(eps_cfg, t, x).prev_sample

        # RePaint blend: outside the mask, force the latent to match
        # (noised original at t_next) so the unmasked region stays identical.
        if i + 1 < len(sub_timesteps):
            t_next = sub_timesteps[i + 1]
            t_next_batched = torch.full((B,), int(t_next.item()), device=device, dtype=torch.long)
            noise_next = torch.randn_like(y_in)
            y_in_at_next = scheduler.add_noise(y_in, noise_next, t_next_batched)
        else:
            # Last step lands at t=0: the "noised" original is just the original.
            y_in_at_next = y_in

        x = mask * x_denoised + (1.0 - mask) * y_in_at_next

    return x.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input",  type=str, required=True,
                        help="Image file or directory of normal frames.")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output dir (default: <out_root>/generated).")
    parser.add_argument("--ckpt",   choices=["best", "last"], default="best")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--strength",  type=float, default=None,
                        help="SDEdit strength in [0, 1]. Higher = more change, less faithfulness.")
    parser.add_argument("--steps",     type=int,   default=None,
                        help="DDIM inference steps (default from config.inference.ddim_steps).")
    parser.add_argument("--seed",      type=int,   default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_output_paths(cfg)
    set_seed(int(args.seed if args.seed is not None else cfg["project"]["seed"]))

    inf_cfg = cfg["inference"]
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else inf_cfg["cfg_scale"])
    strength = float(args.strength if args.strength is not None else inf_cfg["sdedit_strength"])
    num_steps = int(args.steps if args.steps is not None else inf_cfg["ddim_steps"])
    mask_blend_strength = float(inf_cfg.get("mask_blend_strength", 0.9))
    mask_gamma = float(inf_cfg.get("mask_gamma", 1.5))
    save_debug = bool(inf_cfg.get("save_debug", False))
    use_depth_mask = bool(inf_cfg.get("use_depth_mask", True))

    neg_name = str(inf_cfg.get("cfg_negative_class", "normal")).lower()
    if neg_name not in {"normal", "null"}:
        raise ValueError(f"cfg_negative_class must be 'normal' or 'null'; got {neg_name!r}")
    negative_class = CLASS_NORMAL if neg_name == "normal" else CLASS_NULL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Build model + load EMA from checkpoint ----------------------
    image_size = int(cfg["data"]["image_size"])
    unet = build_unet(
        image_size=image_size,
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        layers_per_block=int(cfg["model"]["layers_per_block"]),
        attention_head_dim=int(cfg["model"]["attention_head_dim"]),
        norm_num_groups=int(cfg["model"]["norm_num_groups"]),
        resnet_time_scale_shift=str(cfg["model"]["resnet_time_scale_shift"]),
    ).to(device)

    ema = EMA(unet, decay=float(cfg["train"]["ema_decay"]))

    ckpt_dir = paths["ckpt"]
    ckpt_path = (ckpt_dir / "checkpoint-best.pt") if args.ckpt == "best" else (ckpt_dir / "checkpoint-last.pt")
    if not ckpt_path.exists():
        alt = "last" if args.ckpt == "best" else "best"
        raise FileNotFoundError(
            f"No {args.ckpt} checkpoint at {ckpt_path}. "
            f"Train first or pass --ckpt {alt}."
        )
    state = torch.load(ckpt_path, map_location=device)
    if "ema" in state:
        ema.load_state_dict(state["ema"])
    elif "ema_state_dict" in state:
        ema.load_state_dict(state["ema_state_dict"])
    else:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no EMA weights "
            f"(looked for 'ema' / 'ema_state_dict'). Got keys: {list(state.keys())}"
        )

    model = ema.ema_model.to(device).eval()

    # Best-effort: enable xformers for memory-efficient attention.
    try:
        model.enable_xformers_memory_efficient_attention()
    except Exception:                                          # noqa: BLE001
        pass

    # -------- Build inference scheduler -----------------------------------
    scheduler = DDIMScheduler(
        num_train_timesteps=int(cfg["scheduler"]["num_train_timesteps"]),
        beta_start=float(cfg["scheduler"]["beta_start"]),
        beta_end=float(cfg["scheduler"]["beta_end"]),
        beta_schedule=str(cfg["scheduler"]["beta_schedule"]),
        prediction_type=str(cfg["scheduler"]["prediction_type"]),
    )
    # Move scheduler tensors to device once -- prevents the classic
    # "alphas_cumprod CPU/CUDA device mismatch" inside add_noise() / step().
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if hasattr(scheduler, "final_alpha_cumprod"):
        scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device)

    # -------- I/O ---------------------------------------------------------
    input_path = Path(args.input)
    files = list_inputs(input_path)
    if not files:
        raise RuntimeError(f"No input images found under {input_path}")

    out_dir = Path(args.output) if args.output is not None else paths["generated"]
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[ycldi.infer] device={device}  ckpt={ckpt_path.name}  "
        f"images={len(files)}  cfg_scale={cfg_scale}  strength={strength}  "
        f"steps={num_steps}  neg_class={neg_name}"
    )

    # -------- Generation loop --------------------------------------------
    for f in tqdm(files, desc="generating"):
        rgb = load_image(f, image_size).to(device)             # (1, 3, H, W) in [0, 1]
        Y_in, CbCr = split_ycbcr(rgb)                          # (1, 1, H, W) in [-1,1] / (1, 2, H, W)

        if use_depth_mask:
            depth = depth_from_rgb(rgb)                        # (1, 1, H, W) in [0, 1]
            m_over, m_under = vulnerability_masks(
                Y_in, depth, blur_sigma=16.0, gamma=mask_gamma,
            )
            m_over = (m_over * mask_blend_strength).clamp(0.0, 1.0)
            m_under = (m_under * mask_blend_strength).clamp(0.0, 1.0)
        else:
            depth = torch.zeros_like(Y_in)
            m_over = torch.full_like(Y_in, mask_blend_strength)
            m_under = torch.full_like(Y_in, mask_blend_strength)

        # ----- Overexposure ------------------------------------------------
        y_over = sdedit_with_mask(
            model, scheduler, y_in=Y_in, mask=m_over,
            target_class=CLASS_OVER, negative_class=negative_class,
            cfg_scale=cfg_scale, strength=strength, num_steps=num_steps,
            device=device,
        )
        y_over_final = m_over * y_over + (1.0 - m_over) * Y_in
        rgb_over = merge_ycbcr(y_over_final, CbCr)

        # ----- Underexposure -----------------------------------------------
        y_under = sdedit_with_mask(
            model, scheduler, y_in=Y_in, mask=m_under,
            target_class=CLASS_UNDER, negative_class=negative_class,
            cfg_scale=cfg_scale, strength=strength, num_steps=num_steps,
            device=device,
        )
        y_under_final = m_under * y_under + (1.0 - m_under) * Y_in
        rgb_under = merge_ycbcr(y_under_final, CbCr)

        # ----- Save --------------------------------------------------------
        stem = f.stem
        save_rgb(rgb,        out_dir / f"{stem}_normal.png")
        save_rgb(rgb_over,   out_dir / f"{stem}_overexposed.png")
        save_rgb(rgb_under,  out_dir / f"{stem}_underexposed.png")

        if save_debug:
            save_gray(((Y_in + 1.0) * 0.5), debug_dir / f"{stem}_Y.png")
            save_gray(depth,                debug_dir / f"{stem}_depth.png")
            save_gray(m_over,               debug_dir / f"{stem}_mask_over.png")
            save_gray(m_under,              debug_dir / f"{stem}_mask_under.png")

    print(f"[ycldi.infer] done -> {out_dir}")


if __name__ == "__main__":
    main()

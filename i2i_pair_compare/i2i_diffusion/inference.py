"""Inference script: produce over/under exposed RGB pairs from a normal frame.

Pipeline:
  1. Load normal RGB and pre-computed depth .npy.
  2. Convert RGB → LAB at full resolution (preserve A,B chroma).
  3. Resize L_normal and depth to model resolution.
  4. Diffusion-sample the predicted L_target at model resolution.
  5. (Default) Predict the residual: r = L_pred_lowres - L_normal_lowres,
     resize r to full res with bilinear, add to full-res L_normal. This way
     all original high-frequency texture is preserved exactly.
  6. Recombine (L_target_full, A, B) → RGB and save.

Usage:
    python -m i2i_diffusion.inference \
        --config i2i_diffusion/config.yaml \
        --ckpt   ./runs/i2i_diffusion/ema_last.pt \
        --normal sample_pairs_npy/normal.jpg \
        --depth  sample_pairs_npy/depth/01a291a4-0d39-4ac4-baaa-34c99eaff48a.npy \
        --mode   under \
        --out    out/under_pred.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from postprocess import lab_to_rgb, recombine_L_with_chroma, rgb_to_lab
from scheduler import DDPMScheduler, ddim_sample, ddpm_sample
from unet import UNet


_MODE_TO_LABEL = {"over": 0, "under": 1, "overexposed": 0, "underexposed": 1}


def _resize_2d(arr: np.ndarray, size_hw: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """Resize a (H, W) float32 array via PIL. mode in {'bilinear', 'nearest'}."""
    pil_mode = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    img = Image.fromarray(arr.astype(np.float32), mode="F").resize(
        (size_hw[1], size_hw[0]), pil_mode
    )
    return np.array(img, dtype=np.float32)


def _save_L_as_gray(L: np.ndarray, path: Path) -> None:
    """Save an L channel in [0, 100] as an 8-bit grayscale PNG."""
    arr = np.clip(L.astype(np.float32) * (255.0 / 100.0), 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing config; please retrain or provide --config")
    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=cfg["model"]["base_channels"],
        channel_mult=tuple(cfg["model"]["channel_mult"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        input_resolution=cfg["data"]["resolution"],
    ).to(device)
    state = ck.get("ema") or ck.get("model")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_inputs(normal_path: Path, depth_path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    img = np.array(Image.open(normal_path).convert("RGB"))
    H, W = img.shape[:2]
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != (H, W):
        depth = _resize_2d(depth, (H, W), mode="bilinear")
    depth = np.clip(depth, 0.0, 1.0)
    return img, depth, (H, W)


@torch.no_grad()
def predict_L(
    model: torch.nn.Module,
    cfg: dict,
    L_normal_full: np.ndarray,
    depth_full: np.ndarray,
    mode: str,
    device: torch.device,
    num_steps: int,
    sampler: str = "ddim",
    use_residual: bool = True,
    guidance_scale: float = 1.0,
    seed: int | None = None,
    save_raw_prefix: Optional[Path] = None,
) -> np.ndarray:
    """Run the diffusion model and return predicted L at full resolution."""
    H, W = L_normal_full.shape
    res = int(cfg["data"]["resolution"])

    L_normal_low = _resize_2d(L_normal_full, (res, res), mode="bilinear")
    depth_low = _resize_2d(depth_full, (res, res), mode="bilinear")

    cond_L = (L_normal_low / 50.0 - 1.0).astype(np.float32)
    depth_n = (depth_low * 2.0 - 1.0).astype(np.float32)
    cond = np.stack([cond_L, depth_n], axis=0)[None]  # (1, 2, H, W)
    cond_t = torch.from_numpy(cond).to(device)

    label = _MODE_TO_LABEL[mode]
    y = torch.tensor([label], device=device, dtype=torch.long)

    if seed is not None:
        torch.manual_seed(seed)

    scheduler = DDPMScheduler(
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        schedule=cfg["diffusion"]["schedule"],
    ).to(device)

    shape = (1, 1, res, res)
    if sampler == "ddpm":
        x = ddpm_sample(model, scheduler, cond_t, y, shape, device, guidance_scale=guidance_scale)
    else:
        x = ddim_sample(
            model,
            scheduler,
            cond_t,
            y,
            shape,
            device,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
    x = torch.clamp(x, -1.0, 1.0).cpu().numpy()[0, 0]
    L_pred_low = (x + 1.0) * 50.0  # back to [0, 100]

    # ── Optional: dump exactly what the model saw and produced ────────────
    # Saves four sibling files next to the main output so you can inspect
    # what the diffusion network learned, with no residual or chroma fix-up.
    if save_raw_prefix is not None:
        prefix = Path(save_raw_prefix)
        L_pred_clipped = np.clip(L_pred_low.astype(np.float32), 0.0, 100.0)
        L_cond_clipped = np.clip(L_normal_low.astype(np.float32), 0.0, 100.0)
        _save_L_as_gray(L_pred_clipped, prefix.with_name(prefix.stem + "_raw_pred.png"))
        _save_L_as_gray(L_cond_clipped, prefix.with_name(prefix.stem + "_raw_cond.png"))
        np.save(prefix.with_name(prefix.stem + "_raw_pred.npy"), L_pred_clipped)
        np.save(prefix.with_name(prefix.stem + "_raw_cond.npy"), L_cond_clipped)
        print(f"[i2i] saved raw pred/cond at {res}x{res} next to {prefix.name}")

    if use_residual:
        residual_low = (L_pred_low - L_normal_low).astype(np.float32)
        residual_full = _resize_2d(residual_low, (H, W), mode="bilinear")
        L_pred_full = np.clip(L_normal_full + residual_full, 0.0, 100.0)
    else:
        L_pred_full = _resize_2d(L_pred_low, (H, W), mode="bilinear")
        L_pred_full = np.clip(L_pred_full, 0.0, 100.0)

    return L_pred_full.astype(np.float32)


def run(
    ckpt: str,
    normal: str,
    depth: str,
    out: str,
    mode: str,
    sampler: str = "ddim",
    num_steps: int = 50,
    use_residual: bool = True,
    guidance_scale: float = 1.0,
    save_L_npy: bool = False,
    save_raw: bool = False,
    seed: int | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt, device)

    img_rgb, depth_full, (H, W) = load_inputs(Path(normal), Path(depth))
    lab_full = rgb_to_lab(img_rgb)
    L_normal_full = lab_full[..., 0]
    A = lab_full[..., 1]
    B = lab_full[..., 2]

    out_path = Path(out)
    L_pred_full = predict_L(
        model,
        cfg,
        L_normal_full,
        depth_full,
        mode=mode,
        device=device,
        num_steps=num_steps,
        sampler=sampler,
        use_residual=use_residual,
        guidance_scale=guidance_scale,
        seed=seed,
        save_raw_prefix=out_path if save_raw else None,
    )

    rgb_out = recombine_L_with_chroma(L_pred_full, A, B)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_out).save(out_path)
    print(f"[i2i] saved {out_path}")

    if save_L_npy:
        np.save(out_path.with_suffix(".L.npy"), L_pred_full.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--normal", type=str, required=True, help="Normal RGB image path")
    parser.add_argument("--depth", type=str, required=True, help="Depth .npy path")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=list(_MODE_TO_LABEL.keys()), default="under")
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--no_residual", action="store_true",
                        help="Disable residual upsampling (default ON).")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--save_L_npy", action="store_true",
                        help="Also save predicted L (full-res, [0,100]) as <out>.L.npy.")
    parser.add_argument("--save_raw", action="store_true",
                        help="Also dump model's raw L pred and the input cond L at "
                             "training resolution (PNG + .npy, no residual / no chroma).")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run(
        ckpt=args.ckpt,
        normal=args.normal,
        depth=args.depth,
        out=args.out,
        mode=args.mode,
        sampler=args.sampler,
        num_steps=args.num_steps,
        use_residual=not args.no_residual,
        guidance_scale=args.guidance_scale,
        save_L_npy=args.save_L_npy,
        save_raw=args.save_raw,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

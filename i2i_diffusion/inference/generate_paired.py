"""
inference/generate_paired.py
-----------------------------
Takes a directory of Normal frames and generates a *paired* synthetic
dataset:

    output_dir/
      normal/          copies of (or symlinks to) original Normal frames
      overexposed/     generated overexposed versions (same filename)
      underexposed/    generated underexposed versions (same filename)

Usage
-----
    python inference/generate_paired.py \
        --checkpoint runs/illum_v1/checkpoints/epoch_0100.pt \
        --source_dir data/raw/normal \
        --output_dir data/paired_synthetic \
        --guidance_scale 5.0 \
        --ddim_steps 50 \
        --image_size 256 \
        --batch_size 4

The script uses the EMA weights if available in the checkpoint,
otherwise falls back to the standard weights.

Classifier-free guidance
------------------------
At each denoising step:
    ε̂ = ε_uncond + γ * (ε_cond - ε_uncond)
where γ = guidance_scale.  Higher γ (3–7) = stronger exposure effect.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# make project root importable when called directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset            import NormalOnlyDataset
from models.unet_conditioned import ClassConditionedUNet
from models.controlnet_lite  import ControlNetLite, ControlNetHookContext
from training.noise_scheduler import DiffusionScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1, 1] → PIL RGB image."""
    t = (t.float().cpu() * 0.5 + 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


# ── main generation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_paired_dataset(
    checkpoint:     str,
    source_dir:     str,
    output_dir:     str,
    guidance_scale: float = 5.0,
    ddim_steps:     int   = 50,
    image_size:     int   = 256,
    batch_size:     int   = 4,
    eta:            float = 0.0,   # 0 = deterministic DDIM
    device:         str   = "cuda",
) -> None:

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info(f"Running inference on {dev}")

    # ── load checkpoint ────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=dev)

    # ── build model ────────────────────────────────────────────────────────
    # These defaults must match the training config
    unet = ClassConditionedUNet(
        num_classes     = 2,
        class_embed_dim = 512,
        in_channels     = 4,
        image_size      = image_size,
    ).to(dev).eval()

    # prefer EMA weights
    if "ema" in state:
        log.info("Using EMA weights.")
        unet.load_state_dict(state["ema"])
    else:
        log.info("EMA not found — using regular weights.")
        unet.load_state_dict(state["unet"])

    # ControlNet (optional)
    controlnet = None
    if "controlnet" in state:
        log.info("Loading ControlNet weights.")
        controlnet = ControlNetLite(
            in_channels         = 1,
            block_out_channels  = (128, 256, 512, 512),
            num_layers          = 3,
        ).to(dev).eval()
        controlnet.load_state_dict(state["controlnet"])

    # ── noise scheduler ────────────────────────────────────────────────────
    scheduler = DiffusionScheduler()
    scheduler.set_inference_timesteps(ddim_steps)

    # ── dataset / dataloader ───────────────────────────────────────────────
    ds     = NormalOnlyDataset(source_dir, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = (device == "cuda"),
    )
    log.info(f"Found {len(ds)} normal frames → generating paired triplets.")

    # ── output directories ─────────────────────────────────────────────────
    out_normal = Path(output_dir) / "normal"
    out_over   = Path(output_dir) / "overexposed"
    out_under  = Path(output_dir) / "underexposed"
    for d in [out_normal, out_over, out_under]:
        d.mkdir(parents=True, exist_ok=True)

    # ── per-class labels ───────────────────────────────────────────────────
    # 0 = overexposed,  1 = underexposed  (matches training convention)
    LABEL_OVER  = 0
    LABEL_UNDER = 1

    # ── DDIM sampling loop ─────────────────────────────────────────────────
    def ddim_sample(
        source: torch.Tensor,   # (B, 3, H, W)
        edge:   torch.Tensor,   # (B, 1, H, W)
        label:  int,
    ) -> torch.Tensor:
        """Full DDIM reverse diffusion with CFG."""
        B = source.shape[0]
        labels     = torch.full((B,), label,     device=dev, dtype=torch.long)
        null_labels = torch.full((B,), unet.null_class_idx, device=dev, dtype=torch.long)

        # start from pure noise
        x_t = torch.randn(B, 3, image_size, image_size, device=dev)

        for t in tqdm(
            scheduler.inference_timesteps,
            desc=f"DDIM label={label}",
            leave=False,
        ):
            t_batch = torch.full((B,), t, device=dev, dtype=torch.long)

            # conditioned prediction
            if controlnet is not None:
                res = controlnet(edge)
                with ControlNetHookContext(unet.unet, res):
                    eps_cond = unet(x_t, source, t_batch, labels)
            else:
                eps_cond = unet(x_t, source, t_batch, labels)

            # unconditional prediction
            if controlnet is not None:
                res = controlnet(edge)
                with ControlNetHookContext(unet.unet, res):
                    eps_uncond = unet(x_t, source, t_batch, null_labels)
            else:
                eps_uncond = unet(x_t, source, t_batch, null_labels)

            # classifier-free guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x_t = scheduler.step(eps, t, x_t)

        return x_t.clamp(-1, 1)

    # ── iterate batches ────────────────────────────────────────────────────
    img_counter = 0
    for batch in tqdm(loader, desc="Batches"):
        source = batch["normal"].to(dev)
        edge   = batch["normal_edge"].to(dev)
        paths  = batch["path"]

        gen_over  = ddim_sample(source, edge, LABEL_OVER)
        gen_under = ddim_sample(source, edge, LABEL_UNDER)

        # save individual images
        for i in range(source.shape[0]):
            stem = Path(paths[i]).stem
            fname = f"{stem}.png"

            # save normal copy
            _tensor_to_pil(source[i]).save(out_normal / fname)
            # save generated exposures
            _tensor_to_pil(gen_over[i]).save(out_over / fname)
            _tensor_to_pil(gen_under[i]).save(out_under / fname)

            img_counter += 1

    log.info(
        f"\nDone.  Generated {img_counter} triplets.\n"
        f"  normal/       → {out_normal}\n"
        f"  overexposed/  → {out_over}\n"
        f"  underexposed/ → {out_under}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paired (Normal, Over, Under) dataset from Normal frames."
    )
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--source_dir",     required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--ddim_steps",     type=int,   default=50)
    p.add_argument("--image_size",     type=int,   default=256)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_paired_dataset(**vars(args))

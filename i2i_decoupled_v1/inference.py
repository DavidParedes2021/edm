#!/usr/bin/env python3
# inference.py
"""
Generate Synthetic Paired Dataset from Trained Model.

For each normal frame, generates:
  - {frame_name}_normal.png   (copy of input)
  - {frame_name}_over.png     (synthesized overexposed version)
  - {frame_name}_under.png    (synthesized underexposed version)

Output structure:
  synthetic_dataset/
    normal/   ← ground truth
    over/     ← paired synthetic overexposed
    under/    ← paired synthetic underexposed

Usage:
  python inference.py \
    --checkpoint ./outputs/illumination_ycldi_v1/final/model_final.pt \
    --input_dir  ./data/normal \
    --output_dir ./synthetic_dataset \
    --cfg_scale  7.0 \
    --num_steps  50 \
    --image_size 256
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from utils.device import get_device, to_device, empty_cache
from utils.color_space import (
    rgb_to_ycbcr, ycbcr_to_rgb,
    normalize_luminance, denormalize_luminance, replace_luminance,
)
from utils.diffusion import build_inference_scheduler, ddim_sample
from models.unet import IlluminationUNet

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    model = IlluminationUNet(
        base_channels        = mcfg["base_channels"],
        channel_multipliers  = tuple(mcfg["channel_multipliers"]),
        attention_resolutions= tuple(mcfg["attention_resolutions"]),
        num_res_blocks       = mcfg["num_res_blocks"],
        dropout              = 0.0,  # no dropout at inference
        exposure_embed_dim   = mcfg["exposure_embed_dim"],
        image_size           = dcfg["image_size"],
        use_checkpoint       = False,  # faster inference without checkpointing
    ).to(device)

    # Load EMA weights if available (preferred for inference)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        logger.info("Loaded EMA weights (recommended for inference)")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded training weights (no EMA found)")

    model.eval()
    return model, cfg


@torch.no_grad()
def generate_pair(
    model: IlluminationUNet,
    scheduler,
    normal_rgb: torch.Tensor,
    device: torch.device,
    cfg_scale: float,
    image_size: int,
    amp_dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Generate over and under exposed versions of a normal frame.

    Args:
        normal_rgb: (1, 3, H, W) RGB tensor in [0, 1]

    Returns:
        dict with 'over' and 'under' (1, 3, H, W) RGB tensors in [0, 1]
    """
    normal_rgb = normal_rgb.to(device)

    # Extract luminance (Y channel)
    ycbcr     = rgb_to_ycbcr(normal_rgb)                         # (1, 3, H, W)
    normal_y  = ycbcr[:, :1, :, :]                               # (1, 1, H, W) [0,1]
    normal_yn = normalize_luminance(normal_y).to(amp_dtype)       # (1, 1, H, W) [-1,1]

    outputs = {}
    for label_val, name in [(0, "over"), (1, "under")]:
        label = torch.tensor([label_val], dtype=torch.long, device=device)

        y_gen = ddim_sample(
            model           = model,
            scheduler       = scheduler,
            shape           = normal_yn.shape,
            exposure_labels = label,
            condition_images= normal_yn,
            device          = device,
            cfg_scale       = cfg_scale,
            dtype           = amp_dtype,
        )

        y_gen_01 = denormalize_luminance(y_gen)               # [0, 1]

        # Recombine: generated Y + original Cb, Cr → RGB
        rgb_out = replace_luminance(normal_rgb.float(), y_gen_01.float())
        outputs[name] = rgb_out.clamp(0, 1)

    return outputs


def run_inference(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    cfg_scale: float = 7.0,
    num_steps: int = 50,
    image_size: int = 256,
    batch_size: int = 1,
    use_fp16: bool = False,
):
    device    = get_device()
    amp_dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32

    logger.info(f"Device: {device} | dtype: {amp_dtype}")

    # Load model
    model, cfg = load_model(checkpoint_path, device)
    difcfg     = cfg["diffusion"]

    # Build DDIM scheduler
    scheduler = build_inference_scheduler(
        num_inference_steps  = num_steps,
        num_train_timesteps  = difcfg["num_train_timesteps"],
        beta_schedule        = difcfg["beta_schedule"],
        prediction_type      = difcfg["prediction_type"],
    )

    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    # Output directories
    out = Path(output_dir)
    (out / "normal").mkdir(parents=True, exist_ok=True)
    (out / "over").mkdir(parents=True, exist_ok=True)
    (out / "under").mkdir(parents=True, exist_ok=True)

    # Collect input images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    input_files = sorted([
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in exts
    ])

    if not input_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    logger.info(f"Generating pairs for {len(input_files)} normal frames...")
    logger.info(f"CFG scale: {cfg_scale} | DDIM steps: {num_steps}")

    for img_path in tqdm(input_files, desc="Generating"):
        stem = img_path.stem

        # Load and preprocess
        img = Image.open(img_path).convert("RGB")
        rgb = transform(img).unsqueeze(0)     # (1, 3, H, W)

        # Generate
        with torch.cuda.amp.autocast(enabled=(amp_dtype == torch.float16)):
            pairs = generate_pair(
                model       = model,
                scheduler   = scheduler,
                normal_rgb  = rgb,
                device      = device,
                cfg_scale   = cfg_scale,
                image_size  = image_size,
                amp_dtype   = amp_dtype,
            )

        # Save normal (original, resized)
        save_image(rgb.clamp(0, 1), out / "normal" / f"{stem}.png")

        # Save over and under
        for name in ["over", "under"]:
            save_image(pairs[name], out / name / f"{stem}.png")

        empty_cache()

    logger.info(f"\nDone! Synthetic paired dataset saved to: {output_dir}")
    logger.info(f"  normal/: {len(input_files)} frames (ground truth)")
    logger.info(f"  over/:   {len(input_files)} frames (synthetic)")
    logger.info(f"  under/:  {len(input_files)} frames (synthetic)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic paired illumination dataset")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--input_dir",   type=str, required=True,
                        help="Directory of normal frames")
    parser.add_argument("--output_dir",  type=str, default="./synthetic_dataset")
    parser.add_argument("--cfg_scale",   type=float, default=7.0,
                        help="Classifier-free guidance scale (5–10 recommended)")
    parser.add_argument("--num_steps",   type=int, default=50,
                        help="DDIM inference steps (20 fast, 50 quality, 100 best)")
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--fp16",        action="store_true",
                        help="Use FP16 for faster inference on GPU")
    args = parser.parse_args()

    run_inference(
        checkpoint_path = args.checkpoint,
        input_dir       = args.input_dir,
        output_dir      = args.output_dir,
        cfg_scale       = args.cfg_scale,
        num_steps       = args.num_steps,
        image_size      = args.image_size,
        use_fp16        = args.fp16,
    )


if __name__ == "__main__":
    main()
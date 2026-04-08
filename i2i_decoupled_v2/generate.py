"""
generate.py
------------
Standalone inference script to generate a synthetic paired dataset.

Usage
-----
    python generate.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --normal_dir data/normal \
        --output_dir synthetic_pairs \
        --guidance_scale 5.0 \
        --ddim_steps 50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model.unet import IlluminationUNetV2
from model.diffusion import GaussianDiffusion
from dataset.illumination_dataset import (
    IlluminationDataset,
    rgb_to_lab,
    normalise_L,
    normalise_AB,
    denormalise_L,
    denormalise_AB,
    lab_to_rgb,
)
from utils.misc import EMA, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         type=str, required=True)
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--normal_dir",     type=str, required=True)
    p.add_argument("--output_dir",     type=str, default="./synthetic_pairs")
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--ddim_steps",     type=int,   default=50)
    p.add_argument("--use_ema",        action="store_true", default=True)
    return p.parse_args()


def lab_tensor_to_pil(L_tensor, AB_tensor):
    L_np  = L_tensor[0, 0].cpu().float().numpy()
    AB_np = AB_tensor[0].cpu().float().numpy().transpose(1, 2, 0)
    L_real  = denormalise_L(L_np)
    AB_real = denormalise_AB(AB_np)
    lab = np.concatenate([L_real[:, :, np.newaxis], AB_real], axis=2)
    return Image.fromarray(lab_to_rgb(lab))


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sz     = cfg["data"]["image_size"]
    mcfg   = cfg["model"]
    dcfg   = cfg["diffusion"]
    ncls   = mcfg["num_classes"]

    # Build model
    unet = IlluminationUNetV2(
        image_size            = sz,
        base_channels         = mcfg["base_channels"],
        channel_mult          = tuple(mcfg["channel_mult"]),
        attention_resolutions = tuple(mcfg["attention_resolutions"]),
        num_res_blocks        = mcfg["num_res_blocks"],
        dropout               = 0.0,   # no dropout at inference
        num_classes           = ncls,
    ).to(device)

    diffusion = GaussianDiffusion(
        model         = unet,
        timesteps     = dcfg["timesteps"],
        beta_schedule = dcfg["beta_schedule"],
        device        = device,
    ).to(device)

    ema = EMA(unet, decay=0.9999)
    ckpt = load_checkpoint(Path(args.checkpoint), unet, ema, device=device)

    if args.use_ema:
        ema.apply_shadow(unet)
        logger.info("Using EMA weights for inference")

    diffusion.eval()

    # Output dirs
    out_over  = Path(args.output_dir) / "over"
    out_under = Path(args.output_dir) / "under"
    out_over.mkdir(parents=True, exist_ok=True)
    out_under.mkdir(parents=True, exist_ok=True)

    normal_paths = IlluminationDataset._collect(args.normal_dir)
    logger.info(f"Processing {len(normal_paths)} normal images…")

    for img_path in tqdm(normal_paths):
        img     = Image.open(img_path).convert("RGB").resize((sz, sz), Image.BICUBIC)
        img_np  = np.array(img)
        lab     = rgb_to_lab(img_np)

        L_n  = normalise_L(lab[:, :, 0])
        AB_n = normalise_AB(lab[:, :, 1:])

        L_tensor  = torch.from_numpy(L_n).unsqueeze(0).unsqueeze(0).to(device)
        AB_tensor = torch.from_numpy(AB_n.transpose(2, 0, 1)).unsqueeze(0).to(device)

        stem = Path(img_path).stem

        for cls_id, out_dir in [(0, out_over), (1, out_under)]:
            c     = torch.tensor([cls_id], device=device)
            L_gen = diffusion.ddim_sample(
                cond           = L_tensor,
                c              = c,
                ddim_steps     = args.ddim_steps,
                eta            = 0.0,
                guidance_scale = args.guidance_scale,
                null_class_idx = ncls,
            )
            out_img = lab_tensor_to_pil(L_gen, AB_tensor)
            out_img.save(str(out_dir / f"{stem}.png"))

    if args.use_ema:
        ema.restore(unet)

    logger.info(f"✓ Synthetic pairs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

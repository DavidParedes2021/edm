"""Run inference with a trained illumination-artifact specialist.

Examples
--------
    python sample.py --config config.yaml \
        --artifact overexposure \
        --ckpt outputs/checkpoints/overexposure_best.pt \
        --num_samples 16

    python sample.py --config config.yaml \
        --artifact underexposure \
        --ckpt outputs/checkpoints/underexposure_best.pt \
        --num_samples 16
"""

import argparse
import os

import torch

from illum_diff.config import load_config
from illum_diff.dataset import NormalSampleDataset
from illum_diff.model import build_unet
from illum_diff.sampler import generate_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--artifact", type=str, required=True,
                   choices=["overexposure", "underexposure"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--steps", type=int, default=None,
                   help="Override sample.num_inference_steps.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["model"]["artifact"] = args.artifact
    if args.num_samples is not None: cfg["sample"]["num_samples"]         = int(args.num_samples)
    if args.cfg_scale is not None:   cfg["sample"]["cfg_scale"]           = float(args.cfg_scale)
    if args.steps is not None:       cfg["sample"]["num_inference_steps"] = int(args.steps)

    want_cuda = str(cfg["train"]["device"]).startswith("cuda")
    device = torch.device("cuda") if (want_cuda and torch.cuda.is_available()) else torch.device("cpu")

    model = build_unet(
        image_size=int(cfg["data"]["image_size"]),
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_attn_blocks_from_bottom=int(cfg["model"]["num_attn_blocks_from_bottom"]),
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if "ema" in state:
        model.load_state_dict(state["ema"])     # EMA weights are preferred for sampling.
        print(f"[sample] loaded EMA weights from {args.ckpt}")
    elif "model" in state:
        model.load_state_dict(state["model"])
        print(f"[sample] loaded raw model weights from {args.ckpt}")
    else:
        model.load_state_dict(state)
        print(f"[sample] loaded bare state_dict from {args.ckpt}")
    model.eval()

    normal_ds = NormalSampleDataset(
        img_dir=cfg["data"]["normal_dir"],
        image_size=int(cfg["data"]["image_size"]),
        artifact=args.artifact,
        mask_cfg=cfg["mask"],
        limit=int(cfg["sample"]["num_samples"]),
    )

    out_dir = args.out_dir or os.path.join(cfg["paths"]["samples_dir"], f"final_{args.artifact}")
    generate_samples(cfg, model, device, normal_ds, out_dir)
    print(f"[sample] wrote {len(normal_ds)} sample triplets to {out_dir}")


if __name__ == "__main__":
    main()

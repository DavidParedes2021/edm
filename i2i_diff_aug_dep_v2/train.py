"""Train a single illumination-artifact specialist.

Examples
--------
    # Train the overexposure specialist (default in config.yaml).
    python train.py --config config.yaml --artifact overexposure

    # Train the underexposure specialist with the SAME config file.
    python train.py --config config.yaml --artifact underexposure

    # 4 GB RTX 3050 smoke test (just to confirm the pipeline starts).
    python train.py --config config.yaml --artifact overexposure \
        --batch_size 2 --image_size 192 --num_steps 50
"""

import argparse

from illum_diff.config import load_config
from illum_diff.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--artifact", type=str, default=None,
                   choices=["overexposure", "underexposure"],
                   help="Override cfg.model.artifact.")
    # Useful overrides for the 4 GB smoke test:
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.artifact:        cfg["model"]["artifact"]    = args.artifact
    if args.batch_size:      cfg["train"]["batch_size"]  = int(args.batch_size)
    if args.image_size:      cfg["data"]["image_size"]   = int(args.image_size)
    if args.num_steps:       cfg["train"]["num_steps"]   = int(args.num_steps)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.wandb:           cfg["wandb"]["enabled"]     = True

    Trainer(cfg).train()


if __name__ == "__main__":
    main()

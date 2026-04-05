"""
training/config_utils.py
YAML config loading + merge helpers.
"""

import yaml
import os
import random
import numpy as np
import torch
from pathlib import Path


def load_config(path: str) -> dict:
    """Load YAML config from path."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(mixed_precision: str = "no") -> torch.device:
    """Return cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_dir(cfg: dict) -> str:
    """Create output directory and return its path."""
    out_dir = cfg.get("output_dir", "outputs/run")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Sub-directories
    for sub in ["checkpoints", "samples", "logs"]:
        Path(os.path.join(out_dir, sub)).mkdir(exist_ok=True)
    return out_dir

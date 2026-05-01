"""
Inference: turn a folder of normal frames into their underexposed counterparts,
recombining the generated Y with the original Cb/Cr.

Usage:
    python -m ycldi_under.infer \
        --checkpoint runs/ycldi_under/checkpoints/checkpoint-best/state.pt \
        --input  /path/to/normal_frames \
        --output /path/to/synth_under \
        --target under \
        --cfg-scale 2.5 --steps 50

The output directory mirrors the input filenames. Pairs (input, generated)
are written so they can be loaded as a paired dataset downstream.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from color import (
    rgb_to_ycbcr, ycbcr_to_rgb, replace_y, to_diffusion, from_diffusion
)
from data import _IMG_EXTS, NUM_REAL_CLASSES, CLASS_NORMAL, CLASS_UNDER
from model import ConditionalYUNet
from diffusion import DDPMScheduler, cfg_eps_fn


def list_images(root: Path):
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in _IMG_EXTS)


@torch.no_grad()
def translate_batch(model, sched, rgb01: torch.Tensor, target_cls: int,
                    cfg_scale: float, num_steps: int) -> torch.Tensor:
    """rgb01: (B, 3, H, W) in [0, 1]. Returns (B, 3, H, W) translated RGB in [0, 1]."""
    device = next(model.parameters()).device
    rgb01 = rgb01.to(device)
    ycbcr = rgb_to_ycbcr(rgb01)
    y01  = ycbcr[:, 0:1].clamp(0, 1)
    cbcr = ycbcr[:, 1:].clamp(0, 1)

    y_cond = to_diffusion(y01)
    cls = torch.full((y_cond.shape[0],), target_cls, dtype=torch.long, device=device)
    eps_fn = cfg_eps_fn(model, y_cond, cls, null_cls=NUM_REAL_CLASSES, cfg_scale=cfg_scale)
    y_pm1 = sched.ddim_sample(eps_fn, y_cond.shape, device, num_steps=num_steps)
    y_out01 = from_diffusion(y_pm1)

    ycbcr_out = torch.cat([y_out01, cbcr], dim=1)
    rgb_out = ycbcr_to_rgb(ycbcr_out)
    return rgb_out


def build_model_from_ckpt(state: dict, device) -> tuple[ConditionalYUNet, DDPMScheduler]:
    cfg = state['cfg']
    model = ConditionalYUNet(
        in_channels=2, out_channels=1,
        base_channels=cfg['model']['base_channels'],
        channel_mults=tuple(cfg['model']['channel_mults']),
        num_res_blocks=cfg['model']['num_res_blocks'],
        attention_resolutions=tuple(cfg['model']['attention_resolutions']),
        num_real_classes=NUM_REAL_CLASSES,
        image_size=cfg['data']['image_size'],
        dropout=cfg['model'].get('dropout', 0.0),
    ).to(device)

    # Prefer EMA weights if present.
    sd = state.get('ema', None) or state['model']
    # ema state dict may be missing non-fp params (buffers). Mix into model's sd.
    if 'ema' in state:
        msd = model.state_dict()
        msd.update(sd)
        model.load_state_dict(msd)
    else:
        model.load_state_dict(sd)
    model.eval()

    sched = DDPMScheduler(
        num_train_timesteps=cfg['diffusion']['num_train_timesteps'],
        schedule=cfg['diffusion']['schedule'],
    ).to(device)
    return model, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--target', choices=['under', 'normal'], default='under')
    ap.add_argument('--cfg-scale', type=float, default=2.5)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--image-size', type=int, default=None,
                    help='Override image size (default: use checkpoint config).')
    ap.add_argument('--save-pairs', action='store_true',
                    help='Save side-by-side input|output pairs for inspection.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location=device)
    model, sched = build_model_from_ckpt(state, device)

    img_size = args.image_size or state['cfg']['data']['image_size']
    target_cls = CLASS_UNDER if args.target == 'under' else CLASS_NORMAL

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = list_images(in_root)
    if not paths:
        raise FileNotFoundError(f"no images in {in_root}")
    print(f"found {len(paths)} images, target={args.target}, "
          f"steps={args.steps}, cfg={args.cfg_scale}")

    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    for i in tqdm(range(0, len(paths), args.batch_size)):
        chunk = paths[i:i + args.batch_size]
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk])
        out = translate_batch(model, sched, imgs, target_cls,
                              args.cfg_scale, args.steps).cpu()

        for p_src, x_in, x_out in zip(chunk, imgs, out):
            rel = p_src.relative_to(in_root)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_image(x_out, dst)
            if args.save_pairs:
                pair = torch.cat([x_in, x_out], dim=2)  # H x 2W
                pair_path = dst.with_name(dst.stem + '_pair' + dst.suffix)
                save_image(pair, pair_path)


if __name__ == '__main__':
    main()

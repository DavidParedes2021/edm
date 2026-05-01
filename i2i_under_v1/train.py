"""
Training script for YCLDI underexposure.

Usage:
    python -m ycldi_under.train --config config.yaml

Strategy:
    Self-conditional, class-conditional DDPM on the Y channel.

      target  : y (clean)
      cond    : strongly perturbed copy of y (random gamma + blur + noise),
                forces the model to use the class label to recover the
                target's exposure characteristics rather than copying cond.
      class   : the sample's source folder (0=normal, 1=under), with 10%
                replaced by the null class for CFG.

    At inference (see infer.py), class is set to 1 (under) and y_cond is
    a *clean* normal Y. The CFG signal pushes the output into the under
    distribution while y_cond anchors structure -- which works because the
    perturbation distribution at training time *includes* clean inputs
    (gamma=1, blur=0).
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from color import (
    rgb_to_ycbcr, ycbcr_to_rgb, replace_y, to_diffusion, from_diffusion
)
from data import make_loader, NUM_REAL_CLASSES, CLASS_NORMAL, CLASS_UNDER
from model import ConditionalYUNet
from diffusion import DDPMScheduler, cfg_eps_fn
from losses import YCLDILoss
from checkpoint import CheckpointManager, EMA


# ---------------------------------------------------------------------------
# Condition perturbation. Critical -- without this the model learns identity.
# ---------------------------------------------------------------------------

def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(2.5 * sigma)))
    k = _gaussian_kernel1d(sigma, radius, x.device, x.dtype)
    kx = k.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=x.shape[1])
    x = F.conv2d(x, ky, padding=(radius, 0), groups=x.shape[1])
    return x


def perturb_condition(y01: torch.Tensor) -> torch.Tensor:
    """
    Random photometric corruption of y_cond. Operates in [0, 1].

    The distribution intentionally includes the identity (gamma=1, sigma=0,
    noise_std=0) so the network can also handle clean conditions at
    inference. Random gamma is the dominant component because it teaches
    the model not to read the source's exposure level off y_cond.
    """
    B = y01.shape[0]
    dev = y01.device

    # Random gamma in log-space, ~uniform in [1/2.5, 2.5]
    log_g = torch.empty(B, 1, 1, 1, device=dev).uniform_(-math.log(2.5), math.log(2.5))
    g = log_g.exp()
    out = y01.clamp(1e-4, 1.0).pow(g)

    # Random scale & bias (more exposure jitter)
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.1)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)
    out = (out * s + b).clamp(0.0, 1.0)

    # Random mild blur (per-batch -- cheap)
    sigma = float(torch.empty(1).uniform_(0.0, 1.5).item())
    out = _gauss_blur(out, sigma)

    # Mild Gaussian noise
    n_std = float(torch.empty(1).uniform_(0.0, 0.03).item())
    out = (out + torch.randn_like(out) * n_std).clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Sampling for visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_grid(model, sched, batch, cfg, device, num_steps: int = 50, cfg_scale: float = 2.5):
    """
    Render: rows = (input_normal_rgb, generated_under_rgb, generated_normal_rgb).
    """
    model.eval()
    rgb = batch['rgb'].to(device)
    cbcr = batch['cbcr'].to(device)
    y01 = batch['y'].to(device)
    y_cond = to_diffusion(y01)
    B = y_cond.shape[0]

    null_cls = NUM_REAL_CLASSES

    def gen(target_cls: int):
        cls = torch.full((B,), target_cls, dtype=torch.long, device=device)
        eps_fn = cfg_eps_fn(model, y_cond, cls, null_cls=null_cls, cfg_scale=cfg_scale)
        y_t = sched.ddim_sample(eps_fn, y_cond.shape, device, num_steps=num_steps)
        y_out = from_diffusion(y_t)
        ycbcr_out = replace_y(torch.cat([y_out, cbcr], dim=1), y_out)
        return ycbcr_to_rgb(ycbcr_out)

    rgb_under = gen(CLASS_UNDER)
    rgb_normal = gen(CLASS_NORMAL)

    rows = torch.cat([rgb, rgb_under, rgb_normal], dim=0)
    grid = make_grid(rows.cpu(), nrow=B)
    model.train()
    return grid


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg['output']['base_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.get('seed', 0))

    # ---- data ----
    train_loader = make_loader(cfg, train=True)
    print(f"train samples: {len(train_loader.dataset)}, "
          f"steps/epoch: {len(train_loader)}")

    # Fixed visualization batch (always normal class for repeatability)
    vis_iter = iter(make_loader({**cfg,
                                 'training': {**cfg['training'], 'batch_size': 4}},
                                 train=False))
    vis_batch = next(vis_iter)

    # ---- model ----
    model = ConditionalYUNet(
        in_channels=2,
        out_channels=1,
        base_channels=cfg['model']['base_channels'],
        channel_mults=tuple(cfg['model']['channel_mults']),
        num_res_blocks=cfg['model']['num_res_blocks'],
        attention_resolutions=tuple(cfg['model']['attention_resolutions']),
        num_real_classes=NUM_REAL_CLASSES,
        image_size=cfg['data']['image_size'],
        dropout=cfg['model'].get('dropout', 0.1),
        use_grad_checkpointing=cfg['training'].get('gradient_checkpointing', False),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f} M")

    sched = DDPMScheduler(
        num_train_timesteps=cfg['diffusion']['num_train_timesteps'],
        schedule=cfg['diffusion']['schedule'],
    ).to(device)

    loss_fn = YCLDILoss(
        lambda_eps=cfg['loss']['lambda_eps'],
        lambda_l1=cfg['loss']['lambda_l1'],
        lambda_grad=cfg['loss']['lambda_grad'],
        lambda_vgg=cfg['loss']['lambda_vgg'],
        lambda_hist=cfg['loss']['lambda_hist'],
        use_vgg=cfg['loss'].get('use_vgg', True),
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        betas=(0.9, 0.999),
        weight_decay=cfg['training'].get('weight_decay', 0.0),
    )
    ema = EMA(model, decay=cfg['training']['ema_decay'])
    use_amp = cfg['training']['mixed_precision'] == 'fp16'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt = CheckpointManager(out_dir / 'checkpoints')
    start_epoch = 0
    global_step = 0
    if args.resume:
        state = ckpt.load('last', map_location=device)
        if state is not None:
            model.load_state_dict(state['model'])
            ema.load_state_dict(state['ema'], device)
            optim.load_state_dict(state['optim'])
            scaler.load_state_dict(state['scaler'])
            start_epoch = state['epoch'] + 1
            global_step = state['global_step']
            print(f"resumed from epoch {state['epoch']} step {global_step}")

    cfg_dropout = cfg['training']['cfg_dropout']
    grad_accum = cfg['training'].get('grad_accum_steps', 1)
    log_every = cfg['output'].get('log_every_steps', 50)
    sample_every = cfg['output'].get('sample_every_epochs', 5)
    null_cls = NUM_REAL_CLASSES

    for epoch in range(start_epoch, cfg['training']['num_epochs']):
        t0 = time.time()
        running = {}
        for step, batch in enumerate(train_loader):
            y01 = batch['y'].to(device, non_blocking=True)
            cls = batch['cls'].to(device, non_blocking=True)

            # Build target and perturbed condition
            y_target = to_diffusion(y01)
            y_cond_01 = perturb_condition(y01)
            y_cond = to_diffusion(y_cond_01)

            # CFG dropout
            drop = torch.rand(cls.shape[0], device=device) < cfg_dropout
            cls_in = torch.where(drop, torch.full_like(cls, null_cls), cls)

            # Sample t and noise
            t = torch.randint(0, sched.num_train_timesteps,
                              (y_target.shape[0],), device=device)
            noise = torch.randn_like(y_target)
            y_t = sched.q_sample(y_target, t, noise)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = model(y_t, y_cond, t, cls_in)
                # x0_pred for auxiliary losses (cast to fp32 inside loss)
                x0_pred = sched.predict_x0(y_t, t, eps_pred)
                loss, log = loss_fn(eps_pred, noise, x0_pred, y_target)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1

            for k, v in log.items():
                running[k] = running.get(k, 0.0) + float(v)

            if (step + 1) % log_every == 0:
                avg = {k: v / log_every for k, v in running.items()}
                running = {}
                msg = ' '.join(f"{k}={v:.4f}" for k, v in avg.items())
                print(f"e{epoch} s{step+1}/{len(train_loader)} {msg}")

        dt = time.time() - t0
        print(f"epoch {epoch} done in {dt:.1f}s")

        # Sampling
        if (epoch + 1) % sample_every == 0 or epoch == 0:
            prev = None
            grid = None
            try:
                # Use EMA weights for visualization
                prev = ema.apply_to(model)
                grid = sample_grid(
                    model, sched, vis_batch, cfg, device,
                    num_steps=cfg['inference']['num_inference_steps'],
                    cfg_scale=cfg['inference']['cfg_scale'],
                )
                EMA.restore(model, prev)
                save_image(grid, samples_dir / f"epoch_{epoch:04d}.png")
                print(f"saved sample grid -> {samples_dir / f'epoch_{epoch:04d}.png'}")
            except Exception as e:
                print(f"[warn] sampling failed: {e}")
            finally:
                # Release the EMA weight backup and any GPU tensors held by
                # the sampling pass before the next epoch's backward, otherwise
                # fragmentation from sampling can OOM the first training step.
                del prev, grid
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Checkpoint (use eps_mse as the metric; lower is better)
        metric = avg.get('eps_mse', None) if running == {} else None
        ckpt.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optim': optim.state_dict(),
                'scaler': scaler.state_dict(),
                'cfg': cfg,
            },
            metric=metric,
            metric_lower_better=True,
        )


if __name__ == '__main__':
    main()

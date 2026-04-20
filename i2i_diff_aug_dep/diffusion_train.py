#!/usr/bin/env python3
"""
diffusion_train.py — Train depth-conditioned DDPM on paired L-channel data.

Architecture (Palette-style + depth):
    UNet input channels = 3: [noisy_target_L, source_L, depth]
    UNet output channels = 1: predicted noise on target_L
    Class embedding: {0=overexposed, 1=underexposed}

    Optional depth-FiLM: a small MLP on per-image depth statistics adds to
    the time embedding. Toggle via cfg["model"]["depth_film"].

Usage:
    # Step 0:  compute depth for every normal frame (once)
    python depth_estimator.py --input_dir ./data/normal --output_dir ./data/depth

    # Step 1:  generate paired data
    python generate_pairs.py \\
        --normal_dir ./data/normal --depth_dir ./data/depth \\
        --output_dir ./data/pairs --num_variants 3

    # Step 2:  train
    python diffusion_train.py --config diffusion_config.yaml

    # Step 3:  generate
    python diffusion_inference.py --config diffusion_config.yaml \\
        --checkpoint ./output_diffusion/checkpoints/best.pt
"""

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from PIL import Image
import yaml

from diffusion_dataset import PairedLuminanceDataset, NormalInferenceDataset
from exposure_augment import lab_to_rgb
from losses import (
    SobelEdgeLoss,
    DarknessWeightedL1Loss,
    DepthGradientConsistencyLoss,
)
from chroma_attenuation import attenuate_chroma


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Optional depth-FiLM wrapper
# ═══════════════════════════════════════════════════════════════════════════

class DepthFiLMUNet(nn.Module):
    """Adds a small depth-statistics MLP whose output is injected into the
    UNet's time embedding.

    Implementation detail:
        We reuse the UNet2DModel's `time_embedding` module. At forward time we
        compute a 4-D depth descriptor, project it to time_emb_dim, and add it
        to the standard timestep embedding before the UNet's downsampling.
    """

    def __init__(self, unet: UNet2DModel, depth_feature_dim: int = 4):
        super().__init__()
        self.unet = unet
        # read the time embedding dimensionality from the UNet
        time_emb_dim = unet.time_embedding.linear_2.out_features
        self.depth_mlp = nn.Sequential(
            nn.Linear(depth_feature_dim, 128),
            nn.SiLU(),
            nn.Linear(128, time_emb_dim),
        )
        # zero-init the last layer so training starts identical to no-FiLM
        nn.init.zeros_(self.depth_mlp[-1].weight)
        nn.init.zeros_(self.depth_mlp[-1].bias)

    @staticmethod
    def _depth_stats(d: torch.Tensor) -> torch.Tensor:
        """Per-image summary of the depth channel, (B,) → (B, 4)."""
        B = d.shape[0]
        flat = d.view(B, -1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1)
        p10 = torch.quantile(flat, 0.10, dim=1)
        p90 = torch.quantile(flat, 0.90, dim=1)
        return torch.stack([mean, std, p10, p90], dim=1)

    def forward(self, sample, timestep, class_labels):
        # Depth channel is the 3rd input channel by construction
        depth = sample[:, 2:3]
        stats = self._depth_stats(depth)
        depth_emb = self.depth_mlp(stats)

        # Monkey-patch: intercept the UNet's time_embedding by adding after.
        # Simplest + stable approach: call unet internals manually.
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=sample.device)
        elif t.ndim == 0:
            t = t[None].to(sample.device)
        t = t.expand(sample.shape[0])
        t_emb = self.unet.time_proj(t)
        t_emb = t_emb.to(sample.dtype)
        emb = self.unet.time_embedding(t_emb) + depth_emb

        if self.unet.class_embedding is not None and class_labels is not None:
            class_emb = self.unet.class_embedding(class_labels).to(sample.dtype)
            emb = emb + class_emb

        # down / mid / up path — delegate to the stock UNet
        # (mirrors UNet2DModel.forward without its own time-embedding call)
        skip_sample = sample
        sample = self.unet.conv_in(sample)
        down_block_res_samples = (sample,)
        for down_block in self.unet.down_blocks:
            if hasattr(down_block, "skip_conv"):
                sample, res_samples, skip_sample = down_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.unet.mid_block(sample, emb)
        for up_block in self.unet.up_blocks:
            res = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            if hasattr(up_block, "skip_conv"):
                sample, skip_sample = up_block(sample, res, emb, skip_sample)
            else:
                sample = up_block(hidden_states=sample, res_hidden_states_tuple=res, temb=emb)

        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        class _Out:
            pass
        out = _Out()
        out.sample = sample
        return out


def build_model(cfg: dict, device: torch.device):
    unet = UNet2DModel(
        sample_size=cfg["image"]["size"],
        in_channels=cfg["model"]["in_channels"],    # 3: noisy_L + source_L + depth
        out_channels=cfg["model"]["out_channels"],  # 1
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        layers_per_block=cfg["model"]["layers_per_block"],
        down_block_types=tuple(cfg["model"]["down_block_types"]),
        up_block_types=tuple(cfg["model"]["up_block_types"]),
        num_class_embeds=cfg["model"]["num_class_embeds"],
        attention_head_dim=cfg["model"]["attention_head_dim"],
    )
    if cfg["model"].get("depth_film", False):
        print("[Model] depth-FiLM enabled")
        model = DepthFiLMUNet(unet)
    else:
        model = unet
    model = model.to(device)
    try:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        elif hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
    except (ValueError, AttributeError):
        print("[Model] gradient checkpointing not supported")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Sample generation during training
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_samples(model, scheduler_cfg, inference_cfg, normal_loader,
                     device, epoch, output_dir, image_size,
                     chroma_cfg, num_samples=4):
    """Full DDIM denoising conditioned on source L + depth, then chroma fix."""
    model.eval()
    ddim = DDIMScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
        beta_schedule=scheduler_cfg["beta_schedule"],
        prediction_type=scheduler_cfg["prediction_type"],
    )
    ddim.set_timesteps(inference_cfg["num_inference_steps"], device=device)

    samples_dir = Path(output_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_L, depth, AB, L_orig, depth_full, paths, orig_hw in normal_loader:
        if count >= num_samples:
            break
        source_L = source_L.to(device)
        depth = depth.to(device)
        B = source_L.shape[0]

        for domain_label, domain_name in [(0, "overexposed"), (1, "underexposed")]:
            x = torch.randn_like(source_L)
            cond = torch.full((B,), domain_label, dtype=torch.long, device=device)

            for t in ddim.timesteps:
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)
                model_input = torch.cat([x, source_L, depth], dim=1)
                pred = model(model_input, t_batch, class_labels=cond).sample
                x = ddim.step(pred, t, x).prev_sample

            L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

            for b in range(B):
                if count >= num_samples:
                    break
                H_o, W_o = int(orig_hw[0][b]), int(orig_hw[1][b])
                ab = AB[b].numpy()
                l_o = L_orig[b].numpy()
                d_full = depth_full[b].numpy()

                L_pred_full = np.array(
                    Image.fromarray(L_pred_small[b].astype(np.float32), mode="F").resize(
                        (W_o, H_o), Image.LANCZOS
                    ), dtype=np.float32,
                )
                from scipy.ndimage import gaussian_filter
                sigma = 3.0 * max(H_o, W_o) / 512.0
                L_high = l_o - gaussian_filter(l_o, sigma=sigma)
                L_low_pred = gaussian_filter(L_pred_full, sigma=sigma)
                L_final = np.clip(L_low_pred + L_high, 0.0, 100.0).astype(np.float32)

                # chroma attenuation
                A_new, B_new = attenuate_chroma(
                    ab[..., 0], ab[..., 1],
                    L_new=L_final, L_orig=l_o, depth=d_full,
                    mode=domain_name, cfg=chroma_cfg,
                )
                lab = np.stack([L_final, A_new, B_new], axis=-1)
                rgb = lab_to_rgb(lab)

                fname = f"epoch{epoch:04d}_{Path(paths[b]).stem}_{domain_name}.png"
                Image.fromarray(rgb).save(str(samples_dir / fname))
                count += 1

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: dict, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")
    if device.type == "cuda":
        print(f"        GPU = {torch.cuda.get_device_name(0)}, "
              f"VRAM = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    samp_dir = Path(cfg["output"]["samples_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["logging"]["wandb_project"],
                   entity=cfg["logging"].get("wandb_entity"), config=cfg)

    # ── datasets ───────────────────────────────────────────────────────────
    train_ds = PairedLuminanceDataset(
        pairs_dir=cfg["data"]["pairs_dir"],
        image_size=cfg["image"]["size"],
        augment=True,
        gamma_jitter=cfg["training"].get("gamma_jitter", 0.1),
        depth_noise_sigma=cfg["training"].get("depth_noise_sigma", 0.02),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    normal_ds = NormalInferenceDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
        depth_dir=cfg["data"].get("depth_dir"),
    )
    normal_loader = DataLoader(normal_ds, batch_size=1, shuffle=True, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {param_count / 1e6:.2f} M parameters")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])
    ema = EMAModel(model, decay=cfg["training"]["ema_decay"])

    # ── losses ────────────────────────────────────────────────────────────
    edge_loss_fn = SobelEdgeLoss().to(device)
    dark_loss_fn = DarknessWeightedL1Loss(
        alpha=cfg["losses"].get("dark_alpha", 3.0),
        tau=cfg["losses"].get("dark_tau", 0.3),
    ).to(device)
    dgrad_loss_fn = DepthGradientConsistencyLoss(
        margin=cfg["losses"].get("dgrad_margin", 0.1),
    ).to(device)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    if resume_path and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location="cpu")
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError:
            # allow loading a stock UNet into the FiLM wrapper (or vice versa)
            target = model.unet if hasattr(model, "unet") else model
            target.load_state_dict(ck["model"], strict=False)
        ema.load_state_dict(ck["ema"])
        optimiser.load_state_dict(ck["optimiser"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"]
        best_loss = ck["best_loss"]
        print(f"[Resume] from epoch {start_epoch}, best_loss={best_loss:.5f}")

    # ── training ──────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    mse_w  = cfg["losses"]["mse_weight"]
    edge_w = cfg["losses"]["edge_weight"]
    l1_w   = cfg["losses"]["l1_weight"]
    dark_w = cfg["losses"].get("dark_weight", 0.15)
    dg_w   = cfg["losses"].get("depth_grad_weight", 0.05)
    # SNR gating: auxiliary x0-losses are only meaningful when the x0 estimate
    # has a reasonable signal-to-noise ratio. At high timesteps, x0_hat is
    # dominated by the 1/sqrt(alpha_bar) amplification of pred_noise error
    # and becomes numerically unstable. Default: aux losses fire only on the
    # lower half of the timestep range.
    aux_max_t = int(cfg["losses"].get(
        "aux_max_t_frac", 0.5
    ) * cfg["diffusion"]["num_train_timesteps"])

    # Pre-cached cumulative-alpha tensor (fp32, on device) for the x0 math
    alphas_cumprod_fp32 = noise_scheduler.alphas_cumprod.to(
        device=device, dtype=torch.float32
    )

    nan_skips = 0  # running count of steps skipped due to non-finite loss

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_parts = {"mse": 0.0, "l1": 0.0, "edge": 0.0, "dark": 0.0, "dg": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (source_L, target_L, depth, labels) in enumerate(pbar):
            source_L = source_L.to(device)
            target_L = target_L.to(device)
            depth = depth.to(device)
            labels = labels.to(device)

            noise = torch.randn_like(target_L)
            B = target_L.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()
            noisy_target = noise_scheduler.add_noise(target_L, noise, timesteps)

            # 3-channel input
            model_input = torch.cat([noisy_target, source_L, depth], dim=1)

            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred_noise = model(model_input, timesteps, class_labels=labels).sample
                loss_mse = F.mse_loss(pred_noise, noise)
                total = mse_w * loss_mse

            # ── x0-estimate path (fp32, OUTSIDE autocast) ──────────────────
            # This is the previous NaN source: at high t, alpha_cumprod -> 0,
            # so 1/sqrt(alpha_cumprod) amplifies pred_noise error to fp16-
            # overflow magnitudes. Fix: do the division in fp32, then clamp
            # x0_hat into the valid normalised-L range [-1, 1] before any
            # auxiliary loss sees it, AND gate aux losses off at high t.
            aux_mask = (timesteps <= aux_max_t)           # (B,) bool
            n_aux = int(aux_mask.sum().item())

            parts_this_step = {"mse": float(loss_mse.detach())}

            if n_aux > 0 and (l1_w + edge_w + dark_w + dg_w) > 0:
                pred_noise_f = pred_noise.float()
                noisy_target_f = noisy_target.float()
                target_L_f = target_L.float()
                source_L_f = source_L.float()
                depth_f = depth.float()

                a_bar = alphas_cumprod_fp32[timesteps].view(-1, 1, 1, 1)
                # Floor alpha_bar so sqrt is well above fp16 precision floor.
                a_bar = a_bar.clamp_min(1e-4)
                sqrt_a   = torch.sqrt(a_bar)
                sqrt_1ma = torch.sqrt((1.0 - a_bar).clamp_min(0.0))
                x0_hat = (noisy_target_f - sqrt_1ma * pred_noise_f) / sqrt_a
                # The critical clamp — aux losses only operate on values the
                # *target* could plausibly take.
                x0_hat = x0_hat.clamp(-1.5, 1.5)

                # Select only the subset of the batch that passes the SNR gate
                sel = aux_mask
                x0_s   = x0_hat[sel]
                tgt_s  = target_L_f[sel]
                src_s  = source_L_f[sel]
                dep_s  = depth_f[sel]

                aux_total = 0.0

                if l1_w > 0:
                    l_l1 = F.l1_loss(x0_s, tgt_s)
                    aux_total = aux_total + l1_w * l_l1
                    parts_this_step["l1"] = float(l_l1.detach())

                if edge_w > 0:
                    l_edge = edge_loss_fn(x0_s, tgt_s)
                    aux_total = aux_total + edge_w * l_edge
                    parts_this_step["edge"] = float(l_edge.detach())

                if dark_w > 0:
                    l_dark = dark_loss_fn(x0_s, tgt_s)
                    aux_total = aux_total + dark_w * l_dark
                    parts_this_step["dark"] = float(l_dark.detach())

                if dg_w > 0:
                    l_dg = dgrad_loss_fn(x0_s, src_s, dep_s)
                    aux_total = aux_total + dg_w * l_dg
                    parts_this_step["dg"] = float(l_dg.detach())

                # Scale aux by fraction of batch that passed the gate so its
                # contribution is stable as the SNR-gate selectivity varies.
                total = total + aux_total * (n_aux / B)

            loss = total / grad_accum

            # ── NaN guard ─────────────────────────────────────────────────
            # If any per-part value is non-finite, drop this microbatch
            # rather than contaminating the gradients.
            loss_val = float(loss.detach())
            parts_finite = all(
                (v != v) is False and v not in (float("inf"), float("-inf"))
                for v in parts_this_step.values()
            )
            if (loss_val != loss_val) or loss_val in (float("inf"), float("-inf")) \
                    or not parts_finite:
                nan_skips += 1
                optimiser.zero_grad(set_to_none=True)
                if nan_skips <= 10 or nan_skips % 50 == 0:
                    print(f"[WARN] non-finite loss at epoch {epoch+1} "
                          f"step {step} (total skips: {nan_skips}); "
                          f"parts={parts_this_step}")
                continue

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
                ema.update(model)

            epoch_loss += loss.item() * grad_accum
            for k, v in parts_this_step.items():
                epoch_parts[k] += v
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        avg = epoch_loss / max(n_batches, 1)
        avg_parts = {k: v / max(n_batches, 1) for k, v in epoch_parts.items()}
        print(f"[Epoch {epoch+1}] avg={avg:.5f}  | "
              + " ".join(f"{k}={v:.4f}" for k, v in avg_parts.items())
              + f"  | nan_skips(cumul)={nan_skips}")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "train_loss": avg, **avg_parts})

        # save latest + best
        ck = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
            "config": cfg,
        }
        torch.save(ck, str(ckpt_dir / "latest.pt"))
        if avg < best_loss:
            best_loss = avg
            ck["best_loss"] = best_loss
            torch.save(ck, str(ckpt_dir / "best.pt"))
            print(f"  → best model saved (loss={best_loss:.5f})")

        # periodic samples
        if (epoch + 1) % cfg["training"]["save_every_n_epochs"] == 0:
            orig_sd = copy.deepcopy(model.state_dict())
            ema.apply(model)
            generate_samples(
                model, cfg["diffusion"], cfg["inference"],
                normal_loader, device, epoch + 1, str(samp_dir),
                cfg["image"]["size"],
                chroma_cfg=cfg.get("chroma", {}),
                num_samples=4,
            )
            model.load_state_dict(orig_sd)
            print(f"  → samples saved to {samp_dir}")

    print("[Train] complete.")
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
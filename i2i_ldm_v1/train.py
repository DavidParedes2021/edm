"""
train.py — Main training loop for the Illumination Diffusion LDM pipeline.

Usage:
    python train.py                   # auto-detects GPU tier
    python train.py --smoke           # force smoke-test mode (4 GB GPU)
    python train.py --resume          # resume from checkpoints/last/

Checkpoint strategy  (best / last only — no step_XXXXXXX accumulation):
    checkpoints/
        best/   — saved whenever the 50-step smoothed loss improves
        last/   — overwritten at every save_every interval

Image logging:
    Disk  : samples/step_XXXXXXX_over_loss0.0423.png
             samples/step_XXXXXXX_under_loss0.0423.png
             3-column grid per domain: [Normal | Generated | Target]
    WandB : panels  samples/overexposed  and  samples/underexposed
             with per-image captions (step, EV, domain)
            + full comparison grid under  grids/overexposed  / grids/underexposed
"""

import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from diffusers import DDPMScheduler, DDIMScheduler
from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config  import TrainConfig
from dataset import IlluminationDataset, tensor_to_pil
from model   import (
    EVEmbedding,
    build_unet,
    VAEWrapper,
    LPIPSLoss,
    SSIMLoss,
    HistogramLoss,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Illumination Diffusion Trainer")
    p.add_argument("--smoke",    action="store_true",
                   help="Force smoke-test mode (200 steps, batch=1, 128px)")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from checkpoints/last/ if it exists")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging even if configured")
    p.add_argument("--no-vae",   action="store_true",
                   help="Skip VAE download — pixel-space fallback (dev only)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (best / last only)
# ──────────────────────────────────────────────────────────────────────────────

def _write_ckpt(directory: Path, unet, ev_embed, optimizer, step: int):
    """Write model weights + optimiser state into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(directory / "unet")
    torch.save(ev_embed.state_dict(),  directory / "ev_embed.pt")
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save({"step": step},         directory / "train_state.pt")


def save_last(unet, ev_embed, optimizer, step: int, cfg: TrainConfig):
    """Overwrite checkpoints/last/ — always called at save_every."""
    dest = Path(cfg.output_dir) / "last"
    _write_ckpt(dest, unet, ev_embed, optimizer, step)
    log.info(f"[ckpt] last  saved  (step {step})")


def save_best(unet, ev_embed, optimizer, step: int,
              loss: float, cfg: TrainConfig):
    """Overwrite checkpoints/best/ — only when smoothed loss improved."""
    dest = Path(cfg.output_dir) / "best"
    _write_ckpt(dest, unet, ev_embed, optimizer, step)
    (dest / "best_meta.txt").write_text(f"step={step}\nloss={loss:.6f}\n")
    log.info(f"[ckpt] best  saved  (step {step}, loss {loss:.4f})")


def load_checkpoint(resume_dir: Path, unet, ev_embed, optimizer,
                    cfg: TrainConfig) -> int:
    """Load weights from *resume_dir*, return saved step."""
    from diffusers import UNet2DConditionModel
    state = torch.load(resume_dir / "train_state.pt", map_location="cpu")
    step  = state["step"]
    loaded = UNet2DConditionModel.from_pretrained(resume_dir / "unet")
    unet.load_state_dict(loaded.state_dict())
    del loaded
    ev_embed.load_state_dict(
        torch.load(resume_dir / "ev_embed.pt", map_location=cfg.device)
    )
    optimizer.load_state_dict(
        torch.load(resume_dir / "optimizer.pt", map_location=cfg.device)
    )
    log.info(f"[ckpt] Resumed from {resume_dir}  (step {step})")
    return step


# ──────────────────────────────────────────────────────────────────────────────
# Sample generation + image logging
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    step: int,
    batch: dict,
    unet,
    ev_embed,
    vae,
    ddim_scheduler,
    cfg: TrainConfig,
    accelerator: Accelerator,
    use_wandb: bool,
    best_loss: float,
    n_samples: int = 4,
):
    """
    Run DDIM inference and log sample grids.

    Generates one grid per domain present in the batch:
        columns: [Normal input | Generated output | Pseudo-pair target]

    Disk  : samples/step_XXXXXXX_{domain}_loss{best:.4f}.png
    WandB : samples/{domain}   — list of wandb.Image with EV captions
            grids/{domain}     — full comparison grid as single wandb.Image
    """
    unet.eval()
    ev_embed.eval()

    device = accelerator.device
    dtype  = (
        torch.float16  if cfg.mixed_precision == "fp16"  else
        torch.bfloat16 if cfg.mixed_precision == "bf16"  else
        torch.float32
    )

    out_dir = Path(cfg.samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels  = batch["label"]   # (B,) — 0=over, 1=under
    ev_vals = batch["ev"]

    wandb_log = {}   # accumulated across domains, flushed in one wandb.log

    for domain_label, domain_name in [(0, "over"), (1, "under")]:
        indices = (labels == domain_label).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue

        idx    = indices[:n_samples]
        normal = batch["normal"][idx].to(device)   # (k, 3, H, W)
        target = batch["target"][idx].to(device)
        ev     = ev_vals[idx].to(device)
        k      = normal.shape[0]

        # ── Encode ───────────────────────────────────────────────────────
        if vae is not None:
            normal_latent = vae.encode(normal)
        else:
            h = cfg.image_size // 8
            w = cfg.image_size // 8
            normal_latent = F.interpolate(normal, size=(h, w))

        _, C, H, W = normal_latent.shape
        noisy = torch.randn(k, C, H, W, device=device)

        ev_emb = ev_embed(ev)
        ev_ctx = ev_emb.unsqueeze(1).to(dtype)

        # ── DDIM denoising ───────────────────────────────────────────────
        ddim_scheduler.set_timesteps(cfg.ddim_inference_steps)
        for t in ddim_scheduler.timesteps:
            t_batch  = torch.full((k,), t, device=device, dtype=torch.long)
            mdl_in   = torch.cat([noisy, normal_latent.to(dtype)], dim=1)
            n_pred   = unet(mdl_in, t_batch,
                            encoder_hidden_states=ev_ctx).sample
            noisy    = ddim_scheduler.step(n_pred, t, noisy).prev_sample

        # ── Decode ───────────────────────────────────────────────────────
        if vae is not None:
            generated = vae.decode(noisy.float())
        else:
            generated = F.interpolate(noisy,
                                      size=(cfg.image_size, cfg.image_size))
        generated = generated.clamp(-1, 1)

        # ── 3-column grid: normal | generated | target ───────────────────
        rows = []
        for i in range(k):
            rows += [normal[i].cpu(), generated[i].cpu(), target[i].cpu()]
        grid_01 = (torch.stack(rows).clamp(-1, 1) + 1.0) / 2.0

        # Save to disk
        fname = f"step_{step:07d}_{domain_name}_loss{best_loss:.4f}.png"
        save_image(grid_01, out_dir / fname, nrow=3, padding=2, pad_value=0.5)
        log.info(f"[samples] {domain_name:5s} → {out_dir / fname}")

        # ── WandB image logging ───────────────────────────────────────────
        if use_wandb and HAS_WANDB:
            ev_list = ev.cpu().tolist()

            # Individual generated images with per-image EV caption
            panels = []
            for i in range(k):
                caption = (
                    f"step {step} | EV {ev_list[i]:+.2f} | {domain_name}"
                )
                panels.append(
                    wandb.Image(tensor_to_pil(generated[i].cpu()),
                                caption=caption)
                )
            wandb_log[f"samples/{domain_name}"] = panels

            # Full grid as a single image
            grid_tensor = make_grid(
                grid_01, nrow=3, padding=2, pad_value=0.5
            )
            # grid_tensor is [0,1]; tensor_to_pil expects [-1,1]
            grid_pil = tensor_to_pil(grid_tensor * 2.0 - 1.0)
            wandb_log[f"grids/{domain_name}"] = wandb.Image(
                grid_pil,
                caption=(
                    f"step {step} | {domain_name} | "
                    "normal / generated / target"
                ),
            )

    # One wandb.log call for all domains at this step
    if wandb_log and use_wandb and HAS_WANDB:
        wandb.log(wandb_log, step=step)

    unet.train()
    ev_embed.train()


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    cfg  = TrainConfig()

    # ── Smoke-test overrides ─────────────────────────────────────────────────
    if args.smoke:
        cfg.tier               = "smoke"
        cfg.image_size         = 128
        cfg.train_batch        = 1
        cfg.grad_accum         = 1
        cfg.max_train_steps    = 200
        cfg.save_every         = 100
        cfg.sample_every       = 50
        cfg.mixed_precision    = "fp16" if torch.cuda.is_available() else "no"
        cfg.gradient_checkpointing = True
        log.info("Smoke-test mode: 200 steps, batch=1, 128px")

    use_wandb = cfg.USE_WANDB and HAS_WANDB and not args.no_wandb

    # ── Accelerator ──────────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision             = cfg.mixed_precision,
        gradient_accumulation_steps = cfg.grad_accum,
        log_with                    = "wandb" if use_wandb else None,
        project_dir                 = cfg.log_dir,
    )
    device = accelerator.device
    log.info(f"Device: {device} | tier: {cfg.tier} | "
             f"precision: {cfg.mixed_precision}")

    set_seed(cfg.seed)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    dataset = IlluminationDataset(
        dir_normal       = cfg.data_dir_normal,
        dir_over         = cfg.data_dir_over,
        dir_under        = cfg.data_dir_under,
        image_size       = cfg.image_size,
        domain           = "both",
        ev_over_range    = (cfg.ev_over_min, cfg.ev_over_max),
        ev_under_range   = (cfg.ev_under_min, cfg.ev_under_max),
        use_pseudo_pairs = True,
        augment          = True,
        seed             = cfg.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size         = cfg.train_batch,
        shuffle            = True,
        num_workers        = cfg.dataloader_workers,
        pin_memory         = (device.type == "cuda"),
        drop_last          = True,
        persistent_workers = (cfg.dataloader_workers > 0),
    )

    # ── VAE (frozen) ─────────────────────────────────────────────────────────
    vae = None
    if not args.no_vae and cfg.vae_model_id is not None:
        try:
            vae = VAEWrapper(cfg.vae_model_id,
                             device=str(device),
                             encode_batch=cfg.vae_encode_batch)
        except Exception as e:
            log.warning(f"VAE unavailable ({e}). Pixel-space fallback.")

    # ── Models ───────────────────────────────────────────────────────────────
    ev_embed = EVEmbedding(embed_dim=cfg.ev_embed_dim).to(device)
    unet     = build_unet(
        image_size             = cfg.image_size,
        unet_channels          = cfg.unet_channels,
        unet_layers_per_block  = cfg.unet_layers_per_block,
        ev_embed_dim           = cfg.ev_embed_dim,
        latent_channels        = cfg.latent_channels,
        gradient_checkpointing = cfg.gradient_checkpointing,
    ).to(device)

    # ── Schedulers ───────────────────────────────────────────────────────────
    train_scheduler = DDPMScheduler(
        num_train_timesteps = cfg.num_train_timesteps,
        beta_schedule       = cfg.beta_schedule,
        beta_start          = cfg.beta_start,
        beta_end            = cfg.beta_end,
        clip_sample         = False,
    )
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps = cfg.num_train_timesteps,
        beta_schedule       = cfg.beta_schedule,
        beta_start          = cfg.beta_start,
        beta_end            = cfg.beta_end,
        clip_sample         = False,
    )

    # ── Loss modules ─────────────────────────────────────────────────────────
    lpips_loss = LPIPSLoss(device=str(device))     if cfg.USE_LPIPS else None
    ssim_loss  = SSIMLoss(device=str(device))      if cfg.USE_SSIM  else None
    hist_loss  = HistogramLoss(device=str(device)) if cfg.USE_HIST  else None

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    trainable_params = list(unet.parameters()) + list(ev_embed.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg.learning_rate,
        betas        = (cfg.adam_beta1, cfg.adam_beta2),
        eps          = cfg.adam_eps,
        weight_decay = cfg.adam_weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.lr_warmup_steps:
            return float(step) / max(1, cfg.lr_warmup_steps)
        progress = float(step - cfg.lr_warmup_steps) / max(
            1, cfg.max_train_steps - cfg.lr_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Accelerator.prepare ──────────────────────────────────────────────────
    unet, ev_embed, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, ev_embed, optimizer, dataloader, lr_scheduler
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    last_dir   = Path(cfg.output_dir) / "last"
    if args.resume:
        if last_dir.exists():
            start_step = load_checkpoint(
                last_dir,
                accelerator.unwrap_model(unet),
                accelerator.unwrap_model(ev_embed),
                optimizer, cfg,
            )
        else:
            log.warning("--resume: checkpoints/last/ not found, starting fresh.")

    # ── WandB init ────────────────────────────────────────────────────────────
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.wandb_project,
            config={
                "image_size":     cfg.image_size,
                "eff_batch":      cfg.train_batch * cfg.grad_accum,
                "lr":             cfg.learning_rate,
                "tier":           cfg.tier,
                "max_steps":      cfg.max_train_steps,
                "lambda_lpips":   cfg.lambda_lpips,
                "lambda_ssim":    cfg.lambda_ssim,
                "ev_over_range":  [cfg.ev_over_min, cfg.ev_over_max],
                "ev_under_range": [cfg.ev_under_min, cfg.ev_under_max],
            },
            init_kwargs={"wandb": {"name": cfg.wandb_run_name}},
        )

    # ── Setup dirs ────────────────────────────────────────────────────────────
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.samples_dir).mkdir(parents=True, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    global_step  = start_step
    data_iter    = iter(dataloader)
    last_batch   = None
    best_loss    = float("inf")
    loss_window: list = []
    WINDOW_SIZE  = 50

    log.info(
        f"Training | steps={cfg.max_train_steps} | "
        f"eff_batch={cfg.train_batch * cfg.grad_accum} | "
        f"size={cfg.image_size} | "
        f"vae={'on' if vae else 'off'}"
    )

    unet.train()
    ev_embed.train()

    # ══════════════════════════════════════════════════════════════════════════
    while global_step < cfg.max_train_steps:

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        last_batch = batch

        normal = batch["normal"].to(device)
        target = batch["target"].to(device)
        ev_val = batch["ev"].to(device)

        with accelerator.accumulate(unet):

            # 1. Encode → latents
            if vae is not None:
                with torch.no_grad():
                    normal_latent = vae.encode(normal)
                    target_latent = vae.encode(target)
            else:
                h = cfg.image_size // 8
                with torch.no_grad():
                    normal_latent = F.interpolate(normal, size=(h, h))
                    target_latent = F.interpolate(target, size=(h, h))

            # 2. Add noise
            noise     = torch.randn_like(target_latent)
            B         = target_latent.shape[0]
            timesteps = torch.randint(
                0, cfg.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )
            noisy_target = train_scheduler.add_noise(
                target_latent, noise, timesteps
            )

            # 3. EV embedding
            amp_dtype = (
                torch.float16  if cfg.mixed_precision == "fp16"  else
                torch.bfloat16 if cfg.mixed_precision == "bf16"  else
                torch.float32
            )
            ev_emb = ev_embed(ev_val)
            ev_ctx = ev_emb.unsqueeze(1).to(amp_dtype)

            # 4. Concat Normal latent
            model_input = torch.cat(
                [noisy_target, normal_latent.to(amp_dtype)], dim=1
            )

            # 5. UNet forward
            noise_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=ev_ctx,
            ).sample

            # 6. Diffusion loss
            diff_loss  = F.mse_loss(noise_pred.float(), noise.float())
            total_loss = diff_loss
            loss_dict  = {"loss/diffusion": diff_loss.item()}

            # 7. Auxiliary losses
            if (cfg.USE_LPIPS or cfg.USE_SSIM or cfg.USE_HIST) and vae is not None:
                acp = train_scheduler.alphas_cumprod.to(device)
                sa  = acp[timesteps].sqrt().view(-1, 1, 1, 1)
                som = (1.0 - acp[timesteps]).sqrt().view(-1, 1, 1, 1)
                x0_lat = (noisy_target.float() - som * noise_pred.float()) / (sa + 1e-8)
                with torch.no_grad():
                    x0_px = vae.decode(x0_lat)

                if lpips_loss is not None:
                    lp = lpips_loss(x0_px.to(device), normal.to(device)) * cfg.lambda_lpips
                    total_loss = total_loss + lp
                    loss_dict["loss/lpips"] = lp.item()

                if ssim_loss is not None:
                    ss = ssim_loss(x0_px.to(device), normal.to(device)) * cfg.lambda_ssim
                    total_loss = total_loss + ss
                    loss_dict["loss/ssim"] = ss.item()

                if hist_loss is not None:
                    hl = hist_loss(x0_px.to(device), target.to(device)) * cfg.lambda_hist
                    total_loss = total_loss + hl
                    loss_dict["loss/histogram"] = hl.item()

            loss_dict["loss/total"] = total_loss.item()

            # 8. Backward
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # ── Running loss window ───────────────────────────────────────────────
        loss_window.append(loss_dict["loss/total"])
        if len(loss_window) > WINDOW_SIZE:
            loss_window.pop(0)
        smoothed_loss = sum(loss_window) / len(loss_window)

        # ── Console + scalar WandB logging ───────────────────────────────────
        if global_step % 50 == 0 and accelerator.is_main_process:
            lr_now = lr_scheduler.get_last_lr()[0]
            log.info(
                f"step={global_step:>7d} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                + f" | smooth={smoothed_loss:.4f}"
                + f" | best={best_loss:.4f}"
                + f" | lr={lr_now:.2e}"
            )
            if use_wandb:
                accelerator.log(
                    {**loss_dict,
                     "loss/smoothed": smoothed_loss,
                     "loss/best":     best_loss,
                     "lr":            lr_now},
                    step=global_step,
                )

        # ── Image samples + WandB image logging ──────────────────────────────
        if (
            global_step % cfg.sample_every == 0
            and global_step > 0
            and accelerator.is_main_process
        ):
            generate_samples(
                step           = global_step,
                batch          = last_batch,
                unet           = accelerator.unwrap_model(unet),
                ev_embed       = accelerator.unwrap_model(ev_embed),
                vae            = vae,
                ddim_scheduler = ddim_scheduler,
                cfg            = cfg,
                accelerator    = accelerator,
                use_wandb      = use_wandb,
                best_loss      = best_loss,
            )

        # ── Checkpoints: last always, best on improvement ─────────────────────
        if (
            global_step % cfg.save_every == 0
            and global_step > 0
            and accelerator.is_main_process
        ):
            u = accelerator.unwrap_model(unet)
            e = accelerator.unwrap_model(ev_embed)

            save_last(u, e, optimizer, global_step, cfg)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                save_best(u, e, optimizer, global_step, smoothed_loss, cfg)

        global_step += 1
    # ══════════════════════════════════════════════════════════════════════════

    # Final save
    if accelerator.is_main_process:
        u = accelerator.unwrap_model(unet)
        e = accelerator.unwrap_model(ev_embed)
        save_last(u, e, optimizer, global_step, cfg)
        if smoothed_loss < best_loss:
            save_best(u, e, optimizer, global_step, smoothed_loss, cfg)
        log.info("Training complete.")
        log.info(f"Best smoothed loss : {best_loss:.6f}")
        log.info(f"Checkpoints        : {Path(cfg.output_dir).absolute()}")
        log.info(f"Samples            : {Path(cfg.samples_dir).absolute()}")

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    train()
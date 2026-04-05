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

Key improvements vs original:
  1. Classifier-Free Guidance (CFG) training: EV conditioning is randomly
     replaced by the null embedding with probability cfg_dropout_prob.
     This enables guided inference at any strength (guidance_scale).
  2. EMA: UNet weights are tracked with exponential moving average.
     Samples during training are generated with EMA weights.
  3. Min-SNR weighting: diffusion loss weighted by min(SNR, γ)/SNR.
     Prevents high-noise timesteps from dominating training, accelerating
     convergence significantly.
  4. Auxiliary loss gating: LPIPS/SSIM/histogram losses are only computed
     when t < aux_loss_t_max.  At high t the x0 estimate is extremely noisy
     and these losses add misleading gradients.
  5. LPIPS compared to TARGET (not normal): the perceptual loss now measures
     how well the generated image matches the *exposure target*, not the
     normal input.  SSIM still compares to normal for structure preservation.
"""

import math
import random
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
    EMA,
    build_unet,
    VAEWrapper,
    LPIPSLoss,
    SSIMLoss,
    HistogramLoss,
    ChrominanceConsistencyLoss,
    ExposureBrightnessLoss,
    compute_snr_weights,
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
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_ckpt(directory: Path, unet, ev_embed, optimizer, ema, step: int):
    """Write model weights + optimiser state + EMA into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(directory / "unet")
    torch.save(ev_embed.state_dict(),  directory / "ev_embed.pt")
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save({"step": step},         directory / "train_state.pt")
    if ema is not None:
        torch.save(ema.state_dict(),   directory / "ema.pt")


def save_last(unet, ev_embed, optimizer, ema, step: int, cfg: TrainConfig):
    dest = Path(cfg.output_dir) / "last"
    _write_ckpt(dest, unet, ev_embed, optimizer, ema, step)
    log.info(f"[ckpt] last  saved  (step {step})")


def save_best(unet, ev_embed, optimizer, ema, step: int,
              loss: float, cfg: TrainConfig):
    dest = Path(cfg.output_dir) / "best"
    _write_ckpt(dest, unet, ev_embed, optimizer, ema, step)
    (dest / "best_meta.txt").write_text(f"step={step}\nloss={loss:.6f}\n")
    log.info(f"[ckpt] best  saved  (step {step}, loss {loss:.4f})")


def load_checkpoint(resume_dir: Path, unet, ev_embed, optimizer, ema,
                    cfg: TrainConfig) -> int:
    """Load weights from *resume_dir*, return saved step."""
    from diffusers import UNet2DConditionModel
    state  = torch.load(resume_dir / "train_state.pt", map_location="cpu")
    step   = state["step"]
    loaded = UNet2DConditionModel.from_pretrained(resume_dir / "unet")
    unet.load_state_dict(loaded.state_dict())
    del loaded
    ev_embed.load_state_dict(
        torch.load(resume_dir / "ev_embed.pt", map_location=cfg.device)
    )
    optimizer.load_state_dict(
        torch.load(resume_dir / "optimizer.pt", map_location=cfg.device)
    )
    ema_path = resume_dir / "ema.pt"
    if ema is not None and ema_path.exists():
        ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
        log.info("[ckpt] EMA state restored")
    log.info(f"[ckpt] Resumed from {resume_dir}  (step {step})")
    return step


# ──────────────────────────────────────────────────────────────────────────────
# Sample generation + image logging  (uses EMA weights)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    step: int,
    batch: dict,
    unet,
    ev_embed,
    ema,
    vae,
    ddim_scheduler,
    cfg: TrainConfig,
    accelerator: Accelerator,
    use_wandb: bool,
    best_loss: float,
    n_samples: int = 4,
):
    """
    Run DDIM inference and log sample grids.  Uses EMA weights if available.

    Generates one grid per domain:
        columns: [Normal input | Generated (EMA) | Pseudo-pair target]
    """
    unet.eval()
    ev_embed.eval()

    # Temporarily apply EMA weights for inference
    if ema is not None:
        ema.apply_shadow(unet)

    device = accelerator.device
    dtype  = (
        torch.float16  if cfg.mixed_precision == "fp16"  else
        torch.bfloat16 if cfg.mixed_precision == "bf16"  else
        torch.float32
    )

    out_dir = Path(cfg.samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels  = batch["label"]
    ev_vals = batch["ev"]

    wandb_log = {}

    for domain_label, domain_name in [(0, "over"), (1, "under")]:
        indices = (labels == domain_label).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue

        idx    = indices[:n_samples]
        normal = batch["normal"][idx].to(device)
        target = batch["target"][idx].to(device)
        ev     = ev_vals[idx].to(device)
        k      = normal.shape[0]

        if vae is not None:
            normal_latent = vae.encode(normal)
        else:
            h = cfg.image_size // 8
            normal_latent = F.interpolate(normal, size=(h, h))

        _, C, H, W = normal_latent.shape
        noisy = torch.randn(k, C, H, W, device=device)

        # Conditioned context
        ev_emb = ev_embed(ev)
        ev_ctx = ev_emb.unsqueeze(1).to(dtype)

        # Unconditioned context for CFG
        null_ctx = ev_embed.null_cond(k, device).to(dtype)

        ddim_scheduler.set_timesteps(cfg.ddim_inference_steps)
        for t in ddim_scheduler.timesteps:
            t_batch = torch.full((k,), t, device=device, dtype=torch.long)
            mdl_in  = torch.cat([noisy, normal_latent.to(dtype)], dim=1)

            # Classifier-Free Guidance: two forward passes
            if cfg.guidance_scale > 1.0:
                noise_uncond = unet(mdl_in, t_batch,
                                    encoder_hidden_states=null_ctx).sample
                noise_cond   = unet(mdl_in, t_batch,
                                    encoder_hidden_states=ev_ctx).sample
                # Extrapolate: move in the direction of the conditioning
                n_pred = noise_uncond + cfg.guidance_scale * (noise_cond - noise_uncond)
            else:
                n_pred = unet(mdl_in, t_batch,
                              encoder_hidden_states=ev_ctx).sample

            noisy = ddim_scheduler.step(n_pred, t, noisy).prev_sample

        if vae is not None:
            generated = vae.decode(noisy.float())
        else:
            generated = F.interpolate(noisy, size=(cfg.image_size, cfg.image_size))
        generated = generated.clamp(-1, 1)

        rows = []
        for i in range(k):
            rows += [normal[i].cpu(), generated[i].cpu(), target[i].cpu()]
        grid_01 = (torch.stack(rows).clamp(-1, 1) + 1.0) / 2.0

        fname = f"step_{step:07d}_{domain_name}_loss{best_loss:.4f}.png"
        save_image(grid_01, out_dir / fname, nrow=3, padding=2, pad_value=0.5)
        log.info(f"[samples] {domain_name:5s} → {out_dir / fname}")

        if use_wandb and HAS_WANDB:
            ev_list = ev.cpu().tolist()
            panels  = []
            for i in range(k):
                caption = f"step {step} | EV {ev_list[i]:+.2f} | {domain_name} | EMA"
                panels.append(
                    wandb.Image(tensor_to_pil(generated[i].cpu()), caption=caption)
                )
            wandb_log[f"samples/{domain_name}"] = panels

            grid_tensor = make_grid(grid_01, nrow=3, padding=2, pad_value=0.5)
            grid_pil    = tensor_to_pil(grid_tensor * 2.0 - 1.0)
            wandb_log[f"grids/{domain_name}"] = wandb.Image(
                grid_pil,
                caption=f"step {step} | {domain_name} | normal / generated(EMA) / target",
            )

    if wandb_log and use_wandb and HAS_WANDB:
        wandb.log(wandb_log, step=step)

    # Restore live training weights after inference
    if ema is not None:
        ema.restore(unet)

    unet.train()
    ev_embed.train()


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    cfg  = TrainConfig()

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

    # EMA — initialised from the UNet before accelerator.prepare wraps it
    ema = EMA(unet, decay=cfg.ema_decay) if cfg.use_ema else None

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
    # luminance_only=True: LPIPS measures structural fidelity vs the NORMAL image
    # on the Y channel only, so it does NOT penalise brightness shifts.
    lpips_loss    = LPIPSLoss(device=str(device), luminance_only=True) if cfg.USE_LPIPS    else None
    ssim_loss     = SSIMLoss(device=str(device))                       if cfg.USE_SSIM     else None
    chroma_loss   = ChrominanceConsistencyLoss(device=str(device))     if cfg.USE_CHROMA   else None
    hist_loss     = HistogramLoss(device=str(device))                  if cfg.USE_HIST     else None
    exposure_loss = ExposureBrightnessLoss()                           if cfg.USE_EXPOSURE else None

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    # Include null_embedding in trainable params (it is a Parameter in EVEmbedding)
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
                optimizer, ema, cfg,
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
                "cfg_dropout":    cfg.cfg_dropout_prob,
                "guidance_scale": cfg.guidance_scale,
                "use_ema":        cfg.use_ema,
                "ema_decay":      cfg.ema_decay,
                "snr_gamma":      cfg.snr_gamma if cfg.use_snr_weighting else None,
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
        f"vae={'on' if vae else 'off'} | "
        f"CFG dropout={cfg.cfg_dropout_prob} | "
        f"EMA={'on' if ema else 'off'} | "
        f"Min-SNR={'on' if cfg.use_snr_weighting else 'off'}"
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

            # 2. Sample noise and timesteps
            noise     = torch.randn_like(target_latent)
            B         = target_latent.shape[0]
            timesteps = torch.randint(
                0, cfg.num_train_timesteps, (B,),
                device=device, dtype=torch.long,
            )
            noisy_target = train_scheduler.add_noise(
                target_latent, noise, timesteps
            )

            # 3. EV conditioning — Classifier-Free Guidance training dropout
            amp_dtype = (
                torch.float16  if cfg.mixed_precision == "fp16"  else
                torch.bfloat16 if cfg.mixed_precision == "bf16"  else
                torch.float32
            )

            # For each sample independently, decide whether to use null cond.
            # This is equivalent to training a jointly-conditional and
            # unconditional model, which CFG inference relies on.
            use_null = torch.rand(B, device=device) < cfg.cfg_dropout_prob
            ev_emb   = ev_embed(ev_val)          # (B, ev_embed_dim)
            null_emb = ev_embed.null_embedding.to(device).unsqueeze(0).expand(B, -1)
            # Select real or null embedding per sample in the batch
            cond_emb = torch.where(use_null.unsqueeze(-1), null_emb, ev_emb)
            ev_ctx   = cond_emb.unsqueeze(1).to(amp_dtype)  # (B, 1, ev_embed_dim)

            # 4. Concat Normal latent (spatial conditioning)
            model_input = torch.cat(
                [noisy_target, normal_latent.to(amp_dtype)], dim=1
            )

            # 5. UNet forward
            noise_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=ev_ctx,
            ).sample

            # 6. Diffusion loss (with optional Min-SNR weighting)
            if cfg.use_snr_weighting:
                # Compute per-sample weights and apply element-wise
                snr_weights = compute_snr_weights(
                    train_scheduler, timesteps, gamma=cfg.snr_gamma
                ).to(device)
                per_sample_mse = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="none"
                ).mean(dim=[1, 2, 3])   # (B,)
                diff_loss = (per_sample_mse * snr_weights).mean()
            else:
                diff_loss = F.mse_loss(noise_pred.float(), noise.float())

            total_loss = diff_loss
            loss_dict  = {"loss/diffusion": diff_loss.item()}

            # 7. Auxiliary losses — only when VAE is available and t is low
            # (x0 estimated from high-t noisy latents is dominated by noise)
            low_noise_mask = (timesteps < cfg.aux_loss_t_max)
            apply_aux = (
                vae is not None
                and global_step >= cfg.aux_loss_start_step  # diffusion must stabilise first
                and (cfg.USE_LPIPS or cfg.USE_SSIM or cfg.USE_CHROMA
                     or cfg.USE_HIST or cfg.USE_EXPOSURE)
                and low_noise_mask.any()
            )

            if apply_aux:
                # Estimate x0 (clean latent) from current prediction via
                # the diffusion posterior formula: x0 = (x_t - σ_t * ε_pred) / α_t
                acp = train_scheduler.alphas_cumprod.to(device)
                sa  = acp[timesteps].sqrt().view(-1, 1, 1, 1)
                som = (1.0 - acp[timesteps]).sqrt().view(-1, 1, 1, 1)
                x0_lat = (noisy_target.float() - som * noise_pred.float()) / (sa + 1e-8)

                with torch.no_grad():
                    x0_px = vae.decode(x0_lat)

                # Only average aux losses over low-noise samples in the batch
                mask = low_noise_mask.float()

                if lpips_loss is not None:
                    # Compare to NORMAL on the luminance channel only.
                    # This measures structural/texture fidelity (edges, tissue)
                    # WITHOUT penalising the brightness shift the model must make.
                    # The luminance_only=True flag in LPIPSLoss handles conversion.
                    lp = lpips_loss(x0_px.to(device), normal.to(device))
                    lp = lp * cfg.lambda_lpips
                    total_loss = total_loss + lp
                    loss_dict["loss/lpips"] = lp.item()

                if ssim_loss is not None:
                    # SSIM on luminance vs NORMAL: structural preservation.
                    ss = ssim_loss(x0_px.to(device), normal.to(device))
                    ss = ss * cfg.lambda_ssim
                    total_loss = total_loss + ss
                    loss_dict["loss/ssim"] = ss.item()

                if chroma_loss is not None:
                    # Chrominance (Cb, Cr) vs NORMAL: penalise hue/saturation
                    # shifts.  Real exposure changes affect luminance, not hue.
                    # This is the primary fix for color drift in generated images.
                    ch = chroma_loss(x0_px.to(device), normal.to(device))
                    ch = ch * cfg.lambda_chroma
                    total_loss = total_loss + ch
                    loss_dict["loss/chroma"] = ch.item()

                if hist_loss is not None:
                    hl = hist_loss(x0_px.to(device), target.to(device))
                    hl = hl * cfg.lambda_hist
                    total_loss = total_loss + hl
                    loss_dict["loss/histogram"] = hl.item()

                if exposure_loss is not None:
                    # Penalise when generated brightness direction is wrong
                    el = exposure_loss(x0_px.to(device), normal.to(device), ev_val)
                    el = el * cfg.lambda_exposure
                    total_loss = total_loss + el
                    loss_dict["loss/exposure"] = el.item()

            loss_dict["loss/total"] = total_loss.item()

            # 8. Backward
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # ── EMA update (after each optimiser step) ────────────────────────────
        if ema is not None and accelerator.sync_gradients:
            # Pass global_step so EMA uses adaptive decay (fast early, slow late)
            ema.step(accelerator.unwrap_model(unet), step=global_step)

        # ── Running loss window ───────────────────────────────────────────────
        loss_window.append(loss_dict["loss/total"])
        if len(loss_window) > WINDOW_SIZE:
            loss_window.pop(0)
        smoothed_loss = sum(loss_window) / len(loss_window)

        # ── Notify when aux losses become active ─────────────────────────────
        if global_step == cfg.aux_loss_start_step and accelerator.is_main_process:
            log.info(
                f"[step {global_step}] Auxiliary losses now active "
                f"(LPIPS={cfg.USE_LPIPS}, SSIM={cfg.USE_SSIM}, "
                f"chroma={cfg.USE_CHROMA}, exposure={cfg.USE_EXPOSURE}). "
                f"t_max={cfg.aux_loss_t_max}"
            )

        # ── Console + WandB scalar logging ───────────────────────────────────
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

        # ── Image samples ──────────────────────────────────────────────────────
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
                ema            = ema,
                vae            = vae,
                ddim_scheduler = ddim_scheduler,
                cfg            = cfg,
                accelerator    = accelerator,
                use_wandb      = use_wandb,
                best_loss      = best_loss,
            )

        # ── Checkpoints ────────────────────────────────────────────────────────
        if (
            global_step % cfg.save_every == 0
            and global_step > 0
            and accelerator.is_main_process
        ):
            u = accelerator.unwrap_model(unet)
            e = accelerator.unwrap_model(ev_embed)

            save_last(u, e, optimizer, ema, global_step, cfg)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                save_best(u, e, optimizer, ema, global_step, smoothed_loss, cfg)

        global_step += 1
    # ══════════════════════════════════════════════════════════════════════════

    # Final save
    if accelerator.is_main_process:
        u = accelerator.unwrap_model(unet)
        e = accelerator.unwrap_model(ev_embed)
        save_last(u, e, optimizer, ema, global_step, cfg)
        if smoothed_loss < best_loss:
            save_best(u, e, optimizer, ema, global_step, smoothed_loss, cfg)
        log.info("Training complete.")
        log.info(f"Best smoothed loss : {best_loss:.6f}")
        log.info(f"Checkpoints        : {Path(cfg.output_dir).absolute()}")
        log.info(f"Samples            : {Path(cfg.samples_dir).absolute()}")

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    train()

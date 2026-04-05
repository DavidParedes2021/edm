"""
inference.py — Generate synthetic over/underexposed frames from Normal inputs.

Usage examples:
    # Generate overexposed (+2 EV) from all Normal frames:
    python inference.py --input data/normal --output synthetic/overexposed \
                        --ev 2.0 --checkpoint checkpoints/best

    # Generate underexposed (−2.5 EV) with guidance scale 7:
    python inference.py --input data/normal --output synthetic/underexposed \
                        --ev -2.5 --guidance-scale 7.0 --checkpoint checkpoints/best

    # Sweep multiple EV values (produces a sub-directory per EV):
    python inference.py --input data/normal --output synthetic \
                        --ev-sweep "2.0,2.5,3.0,-1.5,-2.0,-2.5" \
                        --guidance-scale 5.0 --checkpoint checkpoints/best

Guidance scale:
    1.0 — no guidance (equivalent to original unconditional inference)
    3-7 — recommended range for balanced quality / diversity
   >10  — strong conditioning, potential mode collapse / over-saturation
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DDIMScheduler
from config  import TrainConfig
from dataset import list_images, pil_to_tensor, tensor_to_pil
from model   import EVEmbedding, build_unet, VAEWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Illumination Diffusion — Inference")
    p.add_argument("--input",          required=True,
                   help="Directory of Normal input frames.")
    p.add_argument("--output",         required=True,
                   help="Output directory for generated frames.")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint directory (from train.py).")
    p.add_argument("--ev",             type=float, default=2.0,
                   help="Target EV shift (positive=over, negative=under).")
    p.add_argument("--ev-sweep",       type=str, default=None,
                   help="Comma-separated EV values for a sweep run.")
    p.add_argument("--steps",          type=int, default=50,
                   help="DDIM inference steps.")
    p.add_argument("--batch",          type=int, default=1,
                   help="Inference batch size.")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="CFG guidance scale. If not set, uses config default. "
                        "1.0 = no guidance, 5.0 = recommended.")
    p.add_argument("--no-vae",         action="store_true")
    p.add_argument("--use-ema",        action="store_true",
                   help="Load EMA weights if available (ema.pt in checkpoint dir).")
    return p.parse_args()


@torch.no_grad()
def run_inference(
    input_dir: str,
    output_dir: str,
    checkpoint: str,
    ev_target: float,
    ddim_steps: int,
    batch_size: int,
    use_vae: bool,
    guidance_scale: float,
    use_ema: bool,
    cfg: TrainConfig,
):
    device = cfg.device
    ckpt   = Path(checkpoint)
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    unet = build_unet(
        image_size             = cfg.image_size,
        unet_channels          = cfg.unet_channels,
        unet_layers_per_block  = cfg.unet_layers_per_block,
        ev_embed_dim           = cfg.ev_embed_dim,
        latent_channels        = cfg.latent_channels,
        gradient_checkpointing = False,
    ).to(device)

    # Prefer EMA weights for inference (smoother, sharper results)
    ema_path = ckpt / "ema.pt"
    if use_ema and ema_path.exists():
        from model import EMA
        ema = EMA(unet, decay=cfg.ema_decay)
        ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
        ema.apply_shadow(unet)
        print(f"[Inference] Loaded EMA weights from {ema_path}")
    else:
        unet.load_state_dict(
            torch.load(ckpt / "unet" / "pytorch_model.bin", map_location=device)
        )
    unet.eval()

    ev_embed = EVEmbedding(embed_dim=cfg.ev_embed_dim).to(device)
    ev_embed.load_state_dict(
        torch.load(ckpt / "ev_embed.pt", map_location=device)
    )
    ev_embed.eval()

    vae = None
    if use_vae and cfg.vae_model_id:
        try:
            vae = VAEWrapper(cfg.vae_model_id, device=device,
                             encode_batch=cfg.vae_encode_batch)
        except Exception as e:
            print(f"[Warning] VAE unavailable ({e}). Pixel-space fallback.")

    ddim = DDIMScheduler(
        num_train_timesteps = cfg.num_train_timesteps,
        beta_schedule       = cfg.beta_schedule,
        beta_start          = cfg.beta_start,
        beta_end            = cfg.beta_end,
        clip_sample         = False,
    )
    ddim.set_timesteps(ddim_steps)

    # ── Collect inputs ────────────────────────────────────────────────────────
    paths = list_images(input_dir)
    if not paths:
        print(f"No images found in {input_dir!r}")
        return

    use_cfg = guidance_scale > 1.0
    print(f"Generating EV={ev_target:+.1f} | guidance={guidance_scale:.1f} | "
          f"CFG={'on' if use_cfg else 'off'} | "
          f"frames={len(paths)} → {out}")

    dtype = torch.float16 if (cfg.mixed_precision == "fp16"
                              and device == "cuda") else torch.float32

    for i in range(0, len(paths), batch_size):
        chunk   = paths[i : i + batch_size]
        tensors = torch.stack([
            pil_to_tensor(Image.open(p).convert("RGB"), cfg.image_size)
            for p in chunk
        ]).to(device)

        B = tensors.shape[0]

        if vae is not None:
            normal_latent = vae.encode(tensors)
        else:
            h, w = cfg.image_size // 8, cfg.image_size // 8
            normal_latent = F.interpolate(tensors, size=(h, w))

        _, C, H, W = normal_latent.shape
        noisy = torch.randn(B, C, H, W, device=device)

        # Conditioned context
        ev_t   = torch.full((B,), ev_target, device=device, dtype=torch.float32)
        ev_emb = ev_embed(ev_t)
        ev_ctx = ev_emb.unsqueeze(1).to(dtype)

        # Unconditioned context (for CFG)
        null_ctx = ev_embed.null_cond(B, device).to(dtype) if use_cfg else None

        for t in ddim.timesteps:
            t_batch    = torch.full((B,), t, device=device, dtype=torch.long)
            model_in   = torch.cat([noisy, normal_latent.to(dtype)], dim=1)

            if use_cfg:
                # Run unconditional and conditional forward passes
                noise_uncond = unet(model_in, t_batch,
                                    encoder_hidden_states=null_ctx).sample
                noise_cond   = unet(model_in, t_batch,
                                    encoder_hidden_states=ev_ctx).sample
                # Classifier-Free Guidance extrapolation
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(model_in, t_batch,
                                  encoder_hidden_states=ev_ctx).sample

            noisy = ddim.step(noise_pred, t, noisy).prev_sample

        if vae is not None:
            generated = vae.decode(noisy.float())
        else:
            generated = F.interpolate(noisy, size=(cfg.image_size, cfg.image_size))
        generated = generated.clamp(-1, 1)

        for j, src_path in enumerate(chunk):
            out_path = out / src_path.name
            tensor_to_pil(generated[j]).save(out_path)

        print(f"  [{i + len(chunk):>5d}/{len(paths)}] done", end="\r")

    print(f"\nDone. Results in {out}")


def main():
    args = parse_args()
    cfg  = TrainConfig()

    # Allow CLI override of guidance scale; fall back to config default
    guidance = args.guidance_scale if args.guidance_scale is not None else cfg.guidance_scale

    ev_values = (
        [float(v) for v in args.ev_sweep.split(",")]
        if args.ev_sweep
        else [args.ev]
    )

    for ev in ev_values:
        ev_str  = f"{ev:+.1f}".replace("+", "pos").replace("-", "neg")
        out_dir = (
            Path(args.output) / f"ev_{ev_str}"
            if args.ev_sweep
            else args.output
        )
        run_inference(
            input_dir      = args.input,
            output_dir     = str(out_dir),
            checkpoint     = args.checkpoint,
            ev_target      = ev,
            ddim_steps     = args.steps,
            batch_size     = args.batch,
            use_vae        = not args.no_vae,
            guidance_scale = guidance,
            use_ema        = args.use_ema,
            cfg            = cfg,
        )


if __name__ == "__main__":
    main()

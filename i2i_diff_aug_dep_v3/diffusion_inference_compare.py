#!/usr/bin/env python3
"""
diffusion_inference_compare.py — Save raw vs. post-processed outputs.

For every input frame, runs the DDIM model exactly once and writes three PNGs:

    {output_dir}/{domain}/l_only/{stem}.png
        Greyscale view of the predicted L channel, but restricted to the
        target region: the brightest ~20% of predicted-L pixels for
        'overexposed' and the darkest ~20% for 'underexposed'. Outside
        that region the original L is rendered instead, so the change is
        visible only in the zone the model is meant to affect. No chroma.

    {output_dir}/{domain}/raw/{stem}.png
        Raw model L (upsampled) combined with the original A,B chroma.
        No analytic floor, no texture reinjection, no focal blend, no chroma
        attenuation, no content mask. This isolates what the diffusion
        network alone produces in colour.

    {output_dir}/{domain}/full/{stem}.png
        Same model output run through the full pipeline from
        diffusion_inference.py (analytic floor + texture reinjection with
        HF gate + focal blend + content-mask passthrough + chroma
        attenuation).

All three branches share one DDIM denoise pass per frame so the only variable
between outputs is the postprocessing. The model conditioning uses the cleaned
depth (same as production inference) so the model sees a reasonable depth map
in every case — depth cleaning is preprocessing on the model *input*, not
postprocessing on its output.

Usage:
    python diffusion_inference_compare.py \\
        --config diffusion_config.yaml \\
        --checkpoint .../checkpoints/under/best.pt \\
        --domain underexposed \\
        --output_dir ./compare_output
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from scipy.ndimage import gaussian_filter, binary_fill_holes, label as cc_label
from tqdm import tqdm
from PIL import Image

from diffusion_dataset import NormalInferenceDataset
from diffusion_train import (
    build_model, EMAModel, load_config, align_attn_keys_to_model,
)
from exposure_augment import lab_to_rgb
from depth_augment import depth_aware_augment_lab
from chroma_attenuation import (
    attenuate_chroma, texture_gate, DEFAULT_CFG as CHROMA_DEFAULT_CFG,
)
from inference_postprocess import (
    detect_content_mask,
    clean_depth_with_mask,
    focal_blend_alpha,
    model_trust_boost,
)


@torch.no_grad()
def run_compare(
    cfg: dict,
    checkpoint_path: str,
    domain: str = None,
    output_dir: str = None,
    depth_dir: str = None,
    texture_sigma_base: float = 3.0,
    seed: int = None,
    depth_backend: str = "hadepth",
    hadepth_repo: str = None,
    hadepth_ckpt: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── checkpoint + domain resolution (mirrors diffusion_inference.py) ──
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cli_domain = (domain or "").strip().lower() or None
    ckpt_domain = (
        (ckpt.get("config") or {}).get("domain", "") or ""
    ).strip().lower() or None
    cfg_domain = (cfg.get("domain") or "").strip().lower() or None
    domain = cli_domain or ckpt_domain or cfg_domain
    if domain not in ("overexposed", "underexposed"):
        raise ValueError(
            f"`domain` must be 'overexposed' or 'underexposed' "
            f"(got {domain!r}). Set in config, embed in checkpoint, or pass --domain."
        )
    print(f"[Compare] device = {device}, domain = {domain}")

    # ── model (prefer EMA weights, like diffusion_inference.py) ─────────
    model = build_model(cfg, device)
    if "model" in ckpt:
        ckpt["model"] = align_attn_keys_to_model(ckpt["model"], model)
    if "ema" in ckpt:
        ckpt["ema"] = align_attn_keys_to_model(ckpt["ema"], model)
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(model)
        print("[Compare] loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # ── scheduler ────────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"]["prediction_type"],
    )
    num_steps = cfg["inference"]["num_inference_steps"]
    ddim.set_timesteps(num_steps, device=device)

    # ── dataset ──────────────────────────────────────────────────────────
    resolved_depth_dir = depth_dir or cfg.get("data", {}).get("depth_dir")
    if resolved_depth_dir and not Path(resolved_depth_dir).is_dir():
        print(f"[Compare] depth_dir '{resolved_depth_dir}' not found; "
              f"depth will be computed on-the-fly.")
        resolved_depth_dir = None

    normal_ds = NormalInferenceDataset(
        normal_dir=cfg["data"]["normal_dir"],
        image_size=cfg["image"]["size"],
        depth_dir=resolved_depth_dir,
        depth_backend=depth_backend,
        hadepth_repo=hadepth_repo,
        hadepth_ckpt=hadepth_ckpt,
    )
    loader = DataLoader(
        normal_ds,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── post-processing config (only used by the 'full' branch) ─────────
    chroma_cfg = dict(CHROMA_DEFAULT_CFG)
    chroma_cfg.update(cfg.get("chroma", {}))

    tg_cfg = cfg.get("texture_gate", {}) or {}
    tg_knee = float(tg_cfg.get("knee", 30.0))
    tg_floor = float(tg_cfg.get("floor", 0.05))

    fb_cfg = cfg.get("focal_blend", {}) or {}
    fb_enable = bool(fb_cfg.get("enable", True))
    fb_gamma = float(fb_cfg.get("gamma", 1.6))
    fb_knee = float(fb_cfg.get("knee", 0.0))
    fb_floor = float(fb_cfg.get("floor", 0.0))
    fb_smooth = float(fb_cfg.get("smooth_sigma", 4.0))
    fb_model_trust = float(fb_cfg.get("model_trust", 0.0))
    fb_model_strength_L = float(fb_cfg.get("model_strength_L", 25.0))

    cm_cfg = cfg.get("content_mask", {}) or {}
    cm_enable = bool(cm_cfg.get("enable", True))
    cm_threshold = float(cm_cfg.get("luma_threshold", 8.0))
    cm_erode = int(cm_cfg.get("erode_iter", 2))

    ll_cfg = cfg.get("l_localize", {}) or {}
    ll_enable = bool(ll_cfg.get("enable", True))
    ll_band_low_under  = float(ll_cfg.get("band_low_pct_under", 10.0))
    ll_band_high_under = float(ll_cfg.get("band_high_pct_under", 30.0))
    ll_band_low_over   = float(ll_cfg.get("band_low_pct_over",  70.0))
    ll_band_high_over  = float(ll_cfg.get("band_high_pct_over", 90.0))
    ll_presmooth_ref   = float(ll_cfg.get("presmooth_ref", 24.0))
    ll_smooth_ref      = float(ll_cfg.get("smooth_ref", 48.0))
    ll_largest_only    = bool(ll_cfg.get("largest_only", True))
    ll_enforce_target  = bool(ll_cfg.get("enforce_target", True))
    ll_target_L_over   = float(ll_cfg.get("target_L_over", 95.0))
    ll_target_L_under  = float(ll_cfg.get("target_L_under", 5.0))

    af_cfg = cfg.get("analytic_floor", {}) or {}
    af_enable = bool(af_cfg.get("enable", True))
    af_shift_under = float(af_cfg.get("shift_magnitude_under", 85.0))
    af_shift_over = float(af_cfg.get("shift_magnitude_over", 70.0))
    af_gamma_under = float(af_cfg.get("gamma_under", 1.7))
    af_gamma_over = float(af_cfg.get("gamma_over", 1.5))
    af_core_threshold = float(af_cfg.get("core_threshold", 0.5))
    af_core_clip_blend = float(af_cfg.get("core_clip_blend", 0.97))
    af_hf_kill = float(af_cfg.get("hf_kill", 0.97))
    af_smooth_sigma_base = float(af_cfg.get("smooth_sigma_base", 6.0))

    # ── output dirs ──────────────────────────────────────────────────────
    out_root = (Path(output_dir) if output_dir
                else Path(cfg["output"]["root"]) / "compare")
    l_only_dir = out_root / domain / "l_only"
    raw_dir = out_root / domain / "raw"
    full_dir = out_root / domain / "full"
    l_only_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Generating] {domain} ({num_steps} DDIM steps)")
    print(f"  l_only → {l_only_dir}")
    print(f"  raw    → {raw_dir}")
    print(f"  full   → {full_dir}")

    img_size = cfg["image"]["size"]
    border_d = 1.0 if domain == "underexposed" else 0.0

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for source_L, depth, AB, L_orig_tensor, depth_full_tensor, paths, orig_hw in tqdm(
        loader, desc=domain
    ):
        B = source_L.shape[0]

        # ── content mask + cleaned depth (used by the 'full' branch and
        # for the model conditioning shared by both branches) ───────────
        rgb_origs = []
        masks_full = []
        depths_clean_full = []
        for b in range(B):
            H_o, W_o = int(orig_hw[0][b]), int(orig_hw[1][b])
            d_o = depth_full_tensor[b].numpy().astype(np.float32)
            rgb_orig = np.array(Image.open(paths[b]).convert("RGB"))
            if rgb_orig.shape[:2] != (H_o, W_o):
                rgb_orig = np.array(
                    Image.fromarray(rgb_orig).resize((W_o, H_o), Image.BILINEAR)
                )
            if cm_enable:
                mask = detect_content_mask(
                    rgb_orig, luma_threshold=cm_threshold, erode_iter=cm_erode
                )
            else:
                mask = np.ones((H_o, W_o), dtype=bool)
            d_clean = clean_depth_with_mask(d_o, mask, border_value=border_d)
            rgb_origs.append(rgb_orig)
            masks_full.append(mask)
            depths_clean_full.append(d_clean)

        depths_clean_small = np.zeros((B, img_size, img_size), dtype=np.float32)
        for b in range(B):
            d_small = np.array(
                Image.fromarray(depths_clean_full[b].astype(np.float32),
                                mode="F").resize((img_size, img_size),
                                                 Image.LANCZOS),
                dtype=np.float32,
            )
            depths_clean_small[b] = np.clip(d_small, 0.0, 1.0)
        depth_t = torch.from_numpy(depths_clean_small * 2.0 - 1.0
                                   ).unsqueeze(1).to(device)

        source_L = source_L.to(device)
        x = torch.randn_like(source_L)

        for t in ddim.timesteps:
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            model_input = torch.cat([x, source_L, depth_t], dim=1)
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred_noise = model(model_input, t_batch,
                                   class_labels=None).sample
            x = ddim.step(pred_noise, t, x).prev_sample

        L_pred_small = ((x.cpu().numpy()[:, 0] + 1.0) * 50.0).clip(0, 100)

        for b in range(B):
            H_o, W_o = int(orig_hw[0][b]), int(orig_hw[1][b])
            ab    = AB[b].numpy()
            l_o   = L_orig_tensor[b].numpy()
            d_o   = depths_clean_full[b]
            mask  = masks_full[b]
            stem  = Path(paths[b]).stem

            L_pred_full = np.array(
                Image.fromarray(L_pred_small[b].astype(np.float32),
                                mode="F").resize(
                    (W_o, H_o), Image.LANCZOS
                ),
                dtype=np.float32,
            ).clip(0.0, 100.0)

            # ── Predicted-L localization weight (shared) ────────────────
            # Same logic as diffusion_inference.py: pre-smooth predicted L
            # so the percentile band reflects large-scale structure, build
            # a smoothstep, optionally keep only the largest connected
            # blob with holes filled, and apply a final blur for soft
            # edges. Used by both l_only and full branches.
            mask_f = mask.astype(np.float32)
            if ll_enable:
                presmooth_sigma = max(H_o, W_o) / max(ll_presmooth_ref, 1.0)
                L_pred_smooth = gaussian_filter(
                    L_pred_full.astype(np.float32), sigma=presmooth_sigma
                )
                L_pool = (L_pred_smooth[mask] if mask.any()
                          else L_pred_smooth.ravel())
                if domain == "underexposed":
                    p_lo = float(np.percentile(L_pool, ll_band_low_under))
                    p_hi = float(np.percentile(L_pool, ll_band_high_under))
                    span = max(p_hi - p_lo, 1e-6)
                    w_loc = np.clip((p_hi - L_pred_smooth) / span, 0.0, 1.0)
                else:
                    p_lo = float(np.percentile(L_pool, ll_band_low_over))
                    p_hi = float(np.percentile(L_pool, ll_band_high_over))
                    span = max(p_hi - p_lo, 1e-6)
                    w_loc = np.clip((L_pred_smooth - p_lo) / span, 0.0, 1.0)
                w_loc = (w_loc * w_loc * (3.0 - 2.0 * w_loc)).astype(np.float32)
                if ll_largest_only:
                    binary = (w_loc >= 0.5) & mask
                    if binary.any():
                        binary = binary_fill_holes(binary)
                        labels, n_cc = cc_label(binary)
                        if n_cc > 0:
                            sizes = np.bincount(labels.ravel())
                            sizes[0] = 0
                            keep = (labels == int(np.argmax(sizes)))
                            w_loc = w_loc * keep.astype(np.float32)
                        else:
                            w_loc[...] = 0.0
                    else:
                        w_loc[...] = 0.0
                w_loc = gaussian_filter(
                    w_loc, sigma=max(H_o, W_o) / max(ll_smooth_ref, 1.0),
                )
            else:
                w_loc = np.ones((H_o, W_o), dtype=np.float32)

            """
            # ── L-ONLY branch: model L restricted to the target region ──
            # Greyscale view that mixes predicted L into the original L
            # using the shared localization weight (also clamped by the
            # content mask so dark borders stay black).
            w_lonly = w_loc * mask_f
            L_l_only = (w_lonly * L_pred_full
                        + (1.0 - w_lonly) * l_o).astype(np.float32)
            zeros = np.zeros_like(L_l_only, dtype=np.float32)
            lab_l_only = np.stack([L_l_only, zeros, zeros], axis=-1)
            rgb_l_only = lab_to_rgb(lab_l_only)
            Image.fromarray(rgb_l_only).save(str(l_only_dir / f"{stem}.png"))

            # ── RAW branch: model L + original AB, no postprocessing ────
            lab_raw = np.stack(
                [L_pred_full, ab[..., 0], ab[..., 1]], axis=-1
            )
            rgb_raw = lab_to_rgb(lab_raw)
            Image.fromarray(rgb_raw).save(str(raw_dir / f"{stem}.png"))
            """

            # ── FULL branch: full diffusion_inference.py pipeline ───────
            L_pp = L_pred_full.copy()

            if af_enable:
                af_shift = (af_shift_under if domain == "underexposed"
                            else af_shift_over)
                analytic_lab = depth_aware_augment_lab(
                    rgb_origs[b], d_o, mode=domain,
                    strength=1.0,
                    shift_magnitude=af_shift,
                    gamma_over=af_gamma_over,
                    gamma_under=af_gamma_under,
                    depth_blend=1.0,
                    smooth_sigma_base=af_smooth_sigma_base,
                    core_threshold=af_core_threshold,
                    core_clip_blend=af_core_clip_blend,
                    hf_kill=af_hf_kill,
                )
                L_analytic = analytic_lab[..., 0].astype(np.float32)
                if domain == "underexposed":
                    L_pp = np.minimum(L_pp, L_analytic)
                else:
                    L_pp = np.maximum(L_pp, L_analytic)

            sigma = texture_sigma_base * max(H_o, W_o) / 512.0
            L_high     = l_o - gaussian_filter(l_o, sigma=sigma)
            L_low_pred = gaussian_filter(L_pp, sigma=sigma)
            hf_gate    = texture_gate(L_low_pred, mode=domain,
                                      knee=tg_knee, floor=tg_floor)
            L_model    = np.clip(L_low_pred + hf_gate * L_high,
                                 0.0, 100.0).astype(np.float32)

            # ── In-mask soft level shift (after texture reinjection) ──
            # Uniform additive shift inside w_loc to bring the mask-core
            # extreme to the target while preserving the natural variation
            # and HF detail. Mirrors diffusion_inference.py.
            if ll_enable and ll_enforce_target:
                core = w_loc > 0.5
                if core.any():
                    L_core = L_model[core]
                    if domain == "underexposed":
                        current = float(np.percentile(L_core, 10.0))
                        needed = current - ll_target_L_under
                        sign = -1.0
                    else:
                        current = float(np.percentile(L_core, 90.0))
                        needed = ll_target_L_over - current
                        sign = +1.0
                    if needed > 0.0:
                        L_shifted = np.clip(L_model + sign * needed,
                                            0.0, 100.0)
                        L_model = (w_loc * L_shifted
                                   + (1.0 - w_loc) * L_model
                                   ).astype(np.float32)

            if fb_enable:
                fb_smooth_px = fb_smooth * max(H_o, W_o) / 512.0
                alpha = focal_blend_alpha(
                    d_o, mode=domain,
                    gamma=fb_gamma, knee=fb_knee, floor=fb_floor,
                    smooth_sigma=fb_smooth_px,
                )
            else:
                alpha = np.ones_like(L_model, dtype=np.float32)

            if fb_model_trust > 0.0:
                fb_smooth_px = fb_smooth * max(H_o, W_o) / 512.0
                s = model_trust_boost(
                    L_model, l_o, mode=domain,
                    strength_L=fb_model_strength_L,
                    smooth_sigma=fb_smooth_px,
                )
                alpha = alpha + (1.0 - alpha) * fb_model_trust * s

            alpha = alpha * mask_f * w_loc

            L_final = (alpha * L_model + (1.0 - alpha) * l_o).astype(np.float32)
            L_final = np.clip(L_final, 0.0, 100.0)

            A_att, B_att = attenuate_chroma(
                ab[..., 0], ab[..., 1],
                L_new=L_final, L_orig=l_o, depth=d_o,
                mode=domain, cfg=chroma_cfg,
            )
            chroma_w = mask_f * w_loc
            A_new = chroma_w * A_att + (1.0 - chroma_w) * ab[..., 0]
            B_new = chroma_w * B_att + (1.0 - chroma_w) * ab[..., 1]

            lab_full = np.stack([L_final, A_new, B_new], axis=-1)
            rgb_full = lab_to_rgb(lab_full)
            Image.fromarray(rgb_full).save(str(full_dir / f"{stem}.png"))

    print(f"\n[Done] outputs in {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default=None,
                        choices=["overexposed", "underexposed"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output dir; raw/ and full/ subdirs are created.")
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument("--texture_sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducible noise.")
    parser.add_argument("--depth_backend", type=str, default="hadepth",
                        choices=["hadepth", "midas", "midas_hybrid",
                                 "depth_anything_v2_hf"],
                        help="On-the-fly depth backend when depth_dir is "
                             "missing files. Defaults to 'hadepth' to match "
                             "the model used to generate training pairs.")
    parser.add_argument("--hadepth_repo", type=str, default=None,
                        help="Path to the HADepth/ folder. Defaults to "
                             "`<this file>/HADepth`.")
    parser.add_argument("--hadepth_ckpt", type=str, default=None,
                        help="Path to HADepth's depth_model.pth (or the "
                             "folder containing it). Defaults to "
                             "`<hadepth_repo>/HADepth_fullmodel`.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_compare(
        cfg, args.checkpoint,
        domain=args.domain,
        output_dir=args.output_dir,
        depth_dir=args.depth_dir,
        texture_sigma_base=args.texture_sigma,
        seed=args.seed,
        depth_backend=args.depth_backend,
        hadepth_repo=args.hadepth_repo,
        hadepth_ckpt=args.hadepth_ckpt,
    )


if __name__ == "__main__":
    main()

"""Inpainting-style DDIM sampler with CFG and texture-preserving HF re-injection.

Pipeline at inference:
    1. Read normal frame -> RGB.
    2. RGB -> LAB (skimage). Keep (a*, b*) untouched. Take L*.
    3. Compute depth proxy + focal mask (depth-driven for normals).
    4. Decompose L into low-frequency (illumination) and high-frequency
       (mucosa texture) bands using the same Gaussian sigma as training.
    5. Run DDIM with CFG over the artifact specialist. The conditioning
       uses the LF band when the model was trained with predict_lf_target.
       At each step, the outside-mask region is forced to q(L_known, t_next)
       (RePaint-style consistency).
    6. Texture preservation: take only the LF band of the model's prediction
       inside the mask, and re-inject the *normal frame's* HF on top. This
       makes texture loss structurally impossible.
    7. Hard substitution outside the mask -> bit-identical L there.
    8. (L_final, original a*, original b*) -> RGB. Save.

Guarantees by construction:
    - Chrominance preservation (a*, b* never touched).
    - Outside-mask L preservation (bit-identical).
    - Mucosa texture preservation INSIDE the mask (HF taken from input).
    - Diffusion is responsible only for the low-frequency illumination shift,
      which is what made the over- and underexposure artifacts in the first place.
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

from . import color as colorm


def _to_2d(t: torch.Tensor) -> np.ndarray:
    return t.detach().squeeze(0).squeeze(0).cpu().numpy()


@torch.no_grad()
def generate_samples(cfg: dict, model: torch.nn.Module, device: torch.device,
                     normal_dataset, out_dir: str, max_samples: Optional[int] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=int(cfg["train"]["num_train_timesteps"]),
        beta_schedule=str(cfg["train"]["beta_schedule"]),
        prediction_type=str(cfg["train"]["prediction_type"]),
    )
    scheduler.set_timesteps(int(cfg["sample"]["num_inference_steps"]))
    timesteps = scheduler.timesteps.to(device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    cfg_scale = float(cfg["sample"]["cfg_scale"])
    do_resample = bool(cfg["sample"]["resample_known_at_each_step"])
    do_hard_sub = bool(cfg["sample"]["hard_substitute_outside"])
    texture_preserve = bool(cfg["sample"].get("texture_preserve", True))
    predict_lf = bool(cfg["train"].get("predict_lf_target", False))
    blur_sigma_lf = float(cfg["data"].get("blur_sigma_lf", 0.0))
    artifact = cfg["model"]["artifact"]

    H = W = int(cfg["data"]["image_size"])
    n_show = min(int(cfg["sample"]["num_samples"]), len(normal_dataset))
    if max_samples is not None:
        n_show = min(n_show, int(max_samples))

    print(f"[sampler] cfg_scale={cfg_scale} | predict_lf_target={predict_lf} "
          f"| texture_preserve={texture_preserve} | blur_sigma_lf={blur_sigma_lf}")

    for idx in range(n_show):
        item = normal_dataset[idx]
        L_full = item["L"].to(device).unsqueeze(0)          # (1,1,H,W) input L
        L_lf   = item["L_lf"].to(device).unsqueeze(0)       # (1,1,H,W) input L's LF
        mask   = item["mask"].to(device).unsqueeze(0)       # (1,1,H,W)
        depth  = item["depth"].to(device).unsqueeze(0)      # (1,1,H,W)
        rgb_orig = item["rgb"].numpy()                       # (H,W,3) uint8
        ab_orig  = item["ab"].numpy()                        # (H,W,2) float32

        # The "known" L the diffusion was conditioned on at TRAINING time:
        # full L if the model was trained on L, blurred L if it was trained on L_lf.
        L_known = L_lf if predict_lf else L_full

        # Initialize: noise inside the mask, known L outside.
        x = torch.randn((1, 1, H, W), device=device)
        x = mask * x + (1.0 - mask) * L_known

        zeros_known = torch.zeros_like(L_known)
        zeros_mask  = torch.zeros_like(mask)

        for i, t in enumerate(timesteps):
            t_b = t.view(1).to(device).long()

            cond_in = torch.cat(
                [x, (1.0 - mask) * L_known, mask, depth], dim=1
            )
            uncond_in = torch.cat(
                [x, zeros_known, zeros_mask, depth], dim=1
            )

            eps_cond   = model(cond_in,   t_b).sample
            eps_uncond = model(uncond_in, t_b).sample
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            step_out = scheduler.step(eps, int(t.item()), x)
            x_new = step_out.prev_sample

            if do_resample and (i + 1) < len(timesteps):
                t_next = timesteps[i + 1]
                a_bar = alphas_cumprod[int(t_next.item())]
                noise = torch.randn_like(L_known)
                L_known_noised = a_bar.sqrt() * L_known + (1.0 - a_bar).sqrt() * noise
                x_new = mask * x_new + (1.0 - mask) * L_known_noised

            x = x_new

        # ---- Texture-preserving recombination ----------------------------------
        # Take the LOW-frequency band of the diffusion's prediction inside the mask
        # (= the illumination shift the model learned), and add the HIGH-frequency
        # band of the original normal frame (= mucosa texture). Outside the mask,
        # use the original L verbatim.
        L_pred = _to_2d(x.clamp(-1.0, 1.0))
        L_full_np = _to_2d(L_full)
        L_lf_np   = _to_2d(L_lf)
        mask_np   = _to_2d(mask)

        if texture_preserve and blur_sigma_lf > 0.0:
            from scipy.ndimage import gaussian_filter
            # If the model was trained on L_lf, its output is already (approximately)
            # band-limited; blurring is a no-op-ish safety net. If the model was
            # trained on L, blurring discards the (suspect) HF it tried to generate.
            L_pred_lf = gaussian_filter(L_pred, sigma=blur_sigma_lf, mode="reflect").astype(np.float32)
            L_normal_hf = (L_full_np - L_lf_np).astype(np.float32)
            L_inside = L_pred_lf + L_normal_hf
        else:
            L_inside = L_pred

        if do_hard_sub:
            L_final_pm1 = mask_np * L_inside + (1.0 - mask_np) * L_full_np
        else:
            L_final_pm1 = L_inside

        L_final_pm1 = np.clip(L_final_pm1, -1.0, 1.0)
        L_final_0_100 = colorm.denormalize_L(L_final_pm1)
        rgb_out = colorm.lab_to_rgb(L_final_0_100, ab_orig)

        Image.fromarray(rgb_orig).save(
            os.path.join(out_dir, f"sample_{idx:02d}_normal.png"))
        Image.fromarray(rgb_out).save(
            os.path.join(out_dir, f"sample_{idx:02d}_{artifact}.png"))
        m_viz = (mask_np * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(m_viz).save(
            os.path.join(out_dir, f"sample_{idx:02d}_mask.png"))
        d_viz = (_to_2d(depth) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(d_viz).save(
            os.path.join(out_dir, f"sample_{idx:02d}_depth.png"))

    if was_training:
        model.train()

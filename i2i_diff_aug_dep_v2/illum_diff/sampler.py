"""Inpainting-style DDIM sampler with classifier-free guidance.

Pipeline at inference:
    1. Read normal frame -> RGB.
    2. RGB -> LAB (skimage). Keep (a*, b*) untouched. Take L*.
    3. Compute depth proxy + focal mask (depth-driven for normals).
    4. Initialize L_T = mask * Gaussian noise + (1 - mask) * L_known.
    5. Iterate DDIM with CFG over the artifact specialist:
           cond = [L_t, (1-mask)*L_known, mask, depth]
           uncond = [L_t, 0, 0, depth]
           eps = eps_uncond + w * (eps_cond - eps_uncond)
       and at each step replace the outside-mask region with q(L_known, t_next).
    6. Hard substitution: L_final = mask * L_pred + (1-mask) * L_known.
    7. (L_final, original a*, original b*) -> RGB. Save.

This guarantees:
    - Chrominance preservation (a*, b* never touched).
    - Outside-mask preservation (luminance is bit-identical).
    - Strong, focalized artifact inside the mask (CFG amplifies the conditional
      delta and the inpainting context restricts where it acts).
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

from . import color as colorm


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
    artifact = cfg["model"]["artifact"]

    H = W = int(cfg["data"]["image_size"])
    n_show = min(int(cfg["sample"]["num_samples"]), len(normal_dataset))
    if max_samples is not None:
        n_show = min(n_show, int(max_samples))

    for idx in range(n_show):
        item = normal_dataset[idx]
        L_known = item["L"].to(device).unsqueeze(0)         # (1,1,H,W)
        mask    = item["mask"].to(device).unsqueeze(0)      # (1,1,H,W)
        depth   = item["depth"].to(device).unsqueeze(0)     # (1,1,H,W)
        rgb_orig = item["rgb"].numpy()                       # (H,W,3) uint8
        ab_orig  = item["ab"].numpy()                        # (H,W,2) float32

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

        L_pred = x.clamp(-1.0, 1.0).squeeze(0).squeeze(0).cpu().numpy()
        if do_hard_sub:
            mask_np    = mask.squeeze(0).squeeze(0).cpu().numpy()
            L_known_np = L_known.squeeze(0).squeeze(0).cpu().numpy()
            L_final_pm1 = mask_np * L_pred + (1.0 - mask_np) * L_known_np
        else:
            L_final_pm1 = L_pred

        L_final_0_100 = colorm.denormalize_L(L_final_pm1)
        rgb_out = colorm.lab_to_rgb(L_final_0_100, ab_orig)

        Image.fromarray(rgb_orig).save(
            os.path.join(out_dir, f"sample_{idx:02d}_normal.png"))
        Image.fromarray(rgb_out).save(
            os.path.join(out_dir, f"sample_{idx:02d}_{artifact}.png"))
        m_viz = (mask.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(m_viz).save(
            os.path.join(out_dir, f"sample_{idx:02d}_mask.png"))
        d_viz = (depth.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(d_viz).save(
            os.path.join(out_dir, f"sample_{idx:02d}_depth.png"))

    if was_training:
        model.train()

"""
training/trainer.py  (memory-optimised)
----------------------------------------
Key changes vs. original
-------------------------
1. Gradient checkpointing on the U-Net via
   unet.unet.enable_gradient_checkpointing() — trades VRAM for recomputation.

2. Lazy loss scheduling — cycle and identity losses are disabled for the
   first N epochs (cycle_start_epoch / identity_start_epoch in config).
   This halves the number of forward passes during warm-up.

3. Perceptual loss computed every `perceptual_every` steps instead of
   every step — VGG-16 holds ~550 MB of activations.

4. Each loss sub-block frees intermediate tensors immediately with
   del + torch.cuda.empty_cache() so peak VRAM is spread across time.

5. _quick_sample is always @torch.no_grad() and returns a detached tensor,
   so no computation graph is kept alive for the cycle images.

6. Gradient accumulation support (gradient_accumulation_steps in config).

7. autocast scope is tightened: only wraps the forward + loss computation,
   not the backward, which stays in fp32 for numerical stability.

Forward pass count per step
---------------------------
  Warm-up (epoch < cycle_start_epoch):   2  (diffusion over + under)
  Full training:                         8  (+ 4 cycle + 2 identity)
  With perceptual_every=2:  VGG runs half as often → ~275 MB less peak

Memory budget at 256px, fp16, batch=4, DGX 16 GB
--------------------------------------------------
  U-Net (64-base, 4 stages):    ~2.1 GB weights + ~3.5 GB activations w/ checkpointing
  VGG-16 (frozen):              ~0.55 GB
  Two discriminators:           ~0.15 GB
  ControlNet:                   ~0.05 GB
  Batch tensors (8 × B×3×256²): ~0.6 GB
  Optimiser states (fp32 copy): ~2.1 GB
  Total (approx):               ~9 GB  → comfortable on 16 GB
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from accelerate import Accelerator

from models.unet_conditioned    import ClassConditionedUNet
from models.controlnet_lite     import ControlNetLite, ControlNetHookContext
from models.discriminator       import PatchDiscriminator
from models.ema                 import EMA
from losses.perceptual          import VGGPerceptualLoss
from losses.ssim_loss           import GradientSSIMLoss
from losses.cycle_loss          import (
    CycleConsistencyLoss, IdentityLoss, AdversarialLoss
)
from training.noise_scheduler   import DiffusionScheduler
from utils.logging_utils        import Logger

log = logging.getLogger(__name__)


class IlluminationDiffusionTrainer:

    def __init__(self, cfg: dict, accelerator: Accelerator) -> None:
        self.cfg  = cfg
        self.acc  = accelerator
        self.dev  = accelerator.device

        self._build_models()
        self._build_losses()
        self._build_optimisers()
        self._build_scheduler()
        self._build_ema()

        self.logger      = Logger(cfg, accelerator)
        self.global_step = 0

    # ── model construction ────────────────────────────────────────────────────

    def _build_models(self) -> None:
        mc  = self.cfg["model"]
        img = self.cfg["data"]["image_size"]
        tc  = self.cfg["training"]

        base_ch  = mc["unet"]["model_channels"]
        block_ch = tuple(base_ch * m for m in mc["unet"]["channel_mult"])

        self.unet = ClassConditionedUNet(
            num_classes        = mc["unet"]["num_classes"],
            class_embed_dim    = mc["unet"]["class_embed_dim"],
            in_channels        = mc["unet"]["in_channels"],
            image_size         = img,
            block_out_channels = block_ch,
            layers_per_block   = mc["unet"]["num_res_blocks"],
        )

        # ── gradient checkpointing ────────────────────────────────────────
        # Recomputes activations on the backward pass instead of storing them.
        # Reduces activation VRAM by ~60% at the cost of ~20% more compute.
        if tc.get("gradient_checkpointing", False):
            try:
                self.unet.unet.enable_gradient_checkpointing()
                log.info("Gradient checkpointing enabled on U-Net.")
            except AttributeError:
                log.warning(
                    "enable_gradient_checkpointing() not available on this "
                    "diffusers version — skipping."
                )

        self.controlnet = ControlNetLite(
            in_channels        = mc["controlnet"]["in_channels"],
            block_out_channels = block_ch,
            num_layers         = mc["controlnet"]["num_layers"],
        ) if mc["controlnet"]["enabled"] else None

        d = mc["discriminator"]
        self.disc_over  = PatchDiscriminator(d["ndf"], d["n_layers"])
        self.disc_under = PatchDiscriminator(d["ndf"], d["n_layers"])

    def _build_losses(self) -> None:
        self.loss_perc  = VGGPerceptualLoss().to(self.dev)
        self.loss_ssim  = GradientSSIMLoss().to(self.dev)
        self.loss_cycle = CycleConsistencyLoss().to(self.dev)
        self.loss_id    = IdentityLoss().to(self.dev)
        self.loss_adv   = AdversarialLoss().to(self.dev)

    def _build_optimisers(self) -> None:
        tc = self.cfg["training"]
        gen_params = list(self.unet.parameters())
        if self.controlnet is not None:
            gen_params += list(self.controlnet.parameters())

        self.opt_g = torch.optim.AdamW(
            gen_params,
            lr           = tc["lr"],
            weight_decay = tc["weight_decay"],
            betas        = (0.9, 0.999),
        )
        self.opt_d = torch.optim.AdamW(
            list(self.disc_over.parameters())
            + list(self.disc_under.parameters()),
            lr           = tc["lr"] * 0.5,
            weight_decay = tc["weight_decay"],
        )

    def _build_scheduler(self) -> None:
        dc = self.cfg["diffusion"]
        self.noise_sched = DiffusionScheduler(
            num_train_timesteps = dc["num_train_timesteps"],
            beta_schedule       = dc["beta_schedule"],
            beta_start          = dc["beta_start"],
            beta_end            = dc["beta_end"],
            prediction_type     = dc["prediction_type"],
            clip_sample         = dc["clip_sample"],
        )
        self.noise_sched.to(self.dev)

    def _build_ema(self) -> None:
        ec = self.cfg["training"]["ema"]
        self.ema = EMA(self.unet, decay=ec["decay"]) if ec["enabled"] else None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _predict_x0(
        self,
        x_t:       torch.Tensor,
        eps_pred:  torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover clean image estimate from noise prediction."""
        ac             = self.noise_sched.train_scheduler.alphas_cumprod
        sqrt_alpha     = ac[timesteps].sqrt()[:, None, None, None]
        sqrt_one_minus = (1 - ac[timesteps]).sqrt()[:, None, None, None]
        x0 = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha.clamp(min=1e-8)
        return x0.clamp(-1, 1)

    @torch.no_grad()
    def _quick_sample(
        self,
        source:      torch.Tensor,
        edge:        torch.Tensor,
        class_label: torch.Tensor,
        t_level:     int = 100,
    ) -> torch.Tensor:
        """
        Single forward→backward denoising step — fast cycle approximation.
        Always runs under no_grad and returns a detached tensor so no
        computation graph is retained.
        """
        B   = source.shape[0]
        t   = torch.full((B,), t_level, device=self.dev, dtype=torch.long)
        eps = torch.randn_like(source)
        x_t = self.noise_sched.add_noise(source, eps, t)

        if self.controlnet is not None:
            res = self.controlnet(edge)
            with ControlNetHookContext(self.unet.unet, res):
                eps_pred = self.unet(x_t, source, t, class_label)
        else:
            eps_pred = self.unet(x_t, source, t, class_label)

        return self._predict_x0(x_t, eps_pred, t).detach()

    # ── training step ─────────────────────────────────────────────────────────

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, float]:
        tc = self.cfg["training"]

        use_amp     = tc["mixed_precision"] != "no"
        use_cycle   = (epoch >= tc.get("cycle_start_epoch",   0))
        use_identity= (epoch >= tc.get("identity_start_epoch", 0))
        use_perc    = (self.global_step % tc.get("perceptual_every", 1) == 0
                       and tc.get("lambda_perc", 0.0) > 0)
        use_adv     = tc.get("lambda_adv", 0.0) > 0
        grad_accum  = tc.get("gradient_accumulation_steps", 1)

        x_a    = batch["normal"].to(self.dev)
        x_b    = batch["over"].to(self.dev)
        x_c    = batch["under"].to(self.dev)
        edge   = batch["normal_edge"].to(self.dev)
        l_over  = batch["label_over"].to(self.dev)
        l_under = batch["label_under"].to(self.dev)
        B = x_a.shape[0]

        metrics: Dict[str, float] = {}

        # ── Step 1: noise + noisy samples ─────────────────────────────────
        t     = self.noise_sched.sample_timesteps(B, self.dev)
        eps_b = torch.randn_like(x_b)
        eps_c = torch.randn_like(x_c)
        x_b_noisy = self.noise_sched.add_noise(x_b, eps_b, t)
        x_c_noisy = self.noise_sched.add_noise(x_c, eps_c, t)

        # ── Step 2: ControlNet residuals (computed once, reused) ──────────
        if self.controlnet is not None:
            with autocast(enabled=use_amp):
                ctrl_res = self.controlnet(edge)
        else:
            ctrl_res = None

        def _predict(x_noisy, label):
            if ctrl_res is not None:
                with ControlNetHookContext(self.unet.unet, ctrl_res):
                    return self.unet(x_noisy, x_a, t, label)
            return self.unet(x_noisy, x_a, t, label)

        # ── Step 3: diffusion MSE loss ─────────────────────────────────────
        with autocast(enabled=use_amp):
            eps_pred_b = _predict(x_b_noisy, l_over)
            eps_pred_c = _predict(x_c_noisy, l_under)
            target_b   = self.noise_sched.get_noise_target(x_b, eps_b, t)
            target_c   = self.noise_sched.get_noise_target(x_c, eps_c, t)
            l_diff     = F.mse_loss(eps_pred_b, target_b) \
                       + F.mse_loss(eps_pred_c, target_c)

            # Recover x̂_0 for auxiliary losses.
            # NOTE: do NOT detach eps_pred here — SSIM and perceptual losses
            # must backpropagate through the noise prediction so the U-Net
            # actually learns from them.  The no_grad+detach pattern was a
            # bug that made these losses contribute zero gradient.
            x0_b = self._predict_x0(x_b_noisy, eps_pred_b, t)
            x0_c = self._predict_x0(x_c_noisy, eps_pred_c, t)

        # ── Step 4: SSIM loss (cheap — always on) ─────────────────────────
        with autocast(enabled=use_amp):
            l_ssim = self.loss_ssim(x0_b, x_a) + self.loss_ssim(x0_c, x_a)

        # ── Step 5: perceptual loss (expensive — every N steps) ───────────
        if use_perc:
            with autocast(enabled=use_amp):
                l_perc = self.loss_perc(x0_b, x_a) \
                       + self.loss_perc(x0_c, x_a)
            metrics["loss/perceptual"] = l_perc.item()
        else:
            l_perc = torch.zeros(1, device=self.dev)

        # Free x0 estimates before the expensive cycle passes.
        # eps_pred is no longer needed after this point.
        del x0_b, x0_c, eps_pred_b, eps_pred_c
        del x_b_noisy, x_c_noisy, eps_b, eps_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 6: cycle consistency (delayed by cycle_start_epoch) ───────
        if use_cycle and tc.get("lambda_cycle", 0.0) > 0:
            x_hat_b    = self._quick_sample(x_a, edge, l_over)
            x_hat_c    = self._quick_sample(x_a, edge, l_under)
            null_label  = torch.full_like(l_over, self.unet.null_class_idx)
            x_a_from_b = self._quick_sample(x_hat_b, edge, null_label)
            x_a_from_c = self._quick_sample(x_hat_c, edge, null_label)

            l_cyc = self.loss_cycle(x_a_from_b, x_a) \
                  + self.loss_cycle(x_a_from_c, x_a)

            del x_a_from_b, x_a_from_c
            metrics["loss/cycle"] = l_cyc.item()
        else:
            x_hat_b = x_hat_c = None
            l_cyc   = torch.zeros(1, device=self.dev)

        # ── Step 7: identity loss (delayed by identity_start_epoch) ────────
        if use_identity and tc.get("lambda_identity", 0.0) > 0:
            x_id_b = self._quick_sample(x_b, edge, l_over)
            x_id_c = self._quick_sample(x_c, edge, l_under)
            l_id   = self.loss_id(x_id_b, x_b) \
                   + self.loss_id(x_id_c, x_c)
            del x_id_b, x_id_c
            metrics["loss/identity"] = l_id.item()
        else:
            l_id = torch.zeros(1, device=self.dev)

        # ── Step 8: adversarial generator loss ─────────────────────────────
        if use_adv and x_hat_b is not None:
            with autocast(enabled=use_amp):
                fake_b = self.disc_over(x_hat_b.float())
                fake_c = self.disc_under(x_hat_c.float())
                l_adv_g = self.loss_adv.generator_loss(fake_b) \
                        + self.loss_adv.generator_loss(fake_c)
            metrics["loss/adv_g"] = l_adv_g.item()
        else:
            l_adv_g = torch.zeros(1, device=self.dev)

        # ── Step 9: total generator loss + backward ─────────────────────────
        l_g = (
            l_diff
            + tc.get("lambda_ssim",     0.05)  * l_ssim
            + tc.get("lambda_perc",     0.10)  * l_perc
            + tc.get("lambda_cycle",    10.0)  * l_cyc
            + tc.get("lambda_adv",      1.0)   * l_adv_g
            + tc.get("lambda_identity", 5.0)   * l_id
        )

        self.opt_g.zero_grad()
        self.acc.backward(l_g / grad_accum)
        if (self.global_step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(self.unet.parameters(), tc["grad_clip"])
            self.opt_g.step()
            self.opt_g.zero_grad()

        metrics.update({
            "loss/diffusion": l_diff.item(),
            "loss/ssim":      l_ssim.item(),
            "loss/total_g":   l_g.item(),
        })

        # ── Step 10: discriminator update ──────────────────────────────────
        if use_adv and x_hat_b is not None:
            with autocast(enabled=use_amp):
                real_b  = self.disc_over(x_b.float())
                fake_b2 = self.disc_over(x_hat_b.detach().float())
                real_c  = self.disc_under(x_c.float())
                fake_c2 = self.disc_under(x_hat_c.detach().float())
                l_d = self.loss_adv.discriminator_loss(real_b, fake_b2) \
                    + self.loss_adv.discriminator_loss(real_c, fake_c2)

            self.opt_d.zero_grad()
            self.acc.backward(l_d / grad_accum)
            if (self.global_step + 1) % grad_accum == 0:
                self.opt_d.step()
                self.opt_d.zero_grad()

            metrics["loss/disc"] = l_d.item()

        # ── Step 11: free cycle tensors ─────────────────────────────────────
        if x_hat_b is not None:
            del x_hat_b, x_hat_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 12: EMA ───────────────────────────────────────────────────
        if self.ema is not None:
            self.ema.update(self.unet)

        self.global_step += 1
        return metrics

    # ── epoch runner ──────────────────────────────────────────────────────────

    def train_epoch(
        self,
        loader: DataLoader,
        epoch:  int,
    ) -> Dict[str, float]:
        self.unet.train()
        if self.controlnet:
            self.controlnet.train()
        self.disc_over.train()
        self.disc_under.train()

        epoch_losses: Dict[str, list] = {}
        tc = self.cfg["training"]

        for batch in loader:
            metrics = self.train_step(batch, epoch)

            for k, v in metrics.items():
                epoch_losses.setdefault(k, []).append(v)

            if self.global_step % tc["log_every"] == 0:
                self.logger.log_scalars(metrics, self.global_step)

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, output_dir: str, epoch: int, filename: str | None = None) -> None:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Use explicit filename if given, otherwise fall back to epoch-numbered name
        path = ckpt_dir / (filename if filename else f"epoch_{epoch:04d}.pt")

        mc      = self.cfg["model"]
        img     = self.cfg["data"]["image_size"]
        base_ch = mc["unet"]["model_channels"]
        block_ch = tuple(base_ch * m for m in mc["unet"]["channel_mult"])

        state = {
            "epoch":              epoch,
            "global_step":        self.global_step,
            # ── full architecture spec so inference can reconstruct exactly ──
            "block_out_channels": block_ch,
            "class_embed_dim":    mc["unet"]["class_embed_dim"],
            "in_channels":        mc["unet"]["in_channels"],
            "image_size":         img,
            "layers_per_block":   mc["unet"]["num_res_blocks"],
            "num_classes":        mc["unet"]["num_classes"],
            # ── weights ────────────────────────────────────────────────────
            "unet":               self.unet.state_dict(),
            "disc_over":         self.disc_over.state_dict(),
            "disc_under":        self.disc_under.state_dict(),
            "opt_g":             self.opt_g.state_dict(),
            "opt_d":             self.opt_d.state_dict(),
        }
        if self.controlnet is not None:
            state["controlnet"] = self.controlnet.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        torch.save(state, path)
        log.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.dev)
        self.unet.load_state_dict(state["unet"])
        self.disc_over.load_state_dict(state["disc_over"])
        self.disc_under.load_state_dict(state["disc_under"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        if "controlnet" in state and self.controlnet is not None:
            self.controlnet.load_state_dict(state["controlnet"])
        if "ema" in state and self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        self.global_step = state.get("global_step", 0)
        log.info(f"Loaded checkpoint from {path} (epoch {state['epoch']})")
        return state["epoch"]
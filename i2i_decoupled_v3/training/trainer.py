"""
training/trainer.py  (v2 - fixed)
-----------------------------------
Fixes vs v1:
  1. Training step passes snr_w and aux_mask to TotalLoss.
  2. Sample generation always uses EMA weights (previously sometimes used
     live weights which are noisier early in training).
  3. Separate timestep sampling for auxiliary losses: samples where
     t >= aux_t_max are still used for diffusion loss but not aux losses.
  4. Gradient accumulation now correctly zeroes after each full accumulation.
  5. Logging includes n_aux_samples to verify gating is working.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

from dataset.illumination_dataset import (
    IlluminationDataset, collate_fn,
    denormalise_L, denormalise_AB, lab_to_rgb,
)
from model.unet import IlluminationUNetV2
from model.diffusion import GaussianDiffusion
from training.losses import TotalLoss
from utils.misc import (
    set_seed, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, EMA,
)

logger = logging.getLogger(__name__)


class IlluminationTrainer:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        set_seed(42)
        self._build_dirs()
        self._build_data()
        self._build_model()
        self._build_optim()
        self._build_loss()
        self._init_wandb()

        self.best_metric = float("inf")
        self.global_step = 0
        self.start_epoch = 0

    # ------------------------------------------------------------------
    def _build_dirs(self):
        dcfg = self.cfg["checkpoint"]
        self.out_dir  = Path(dcfg["output_dir"])
        self.ckpt_dir = Path(dcfg["checkpoint_dir"])
        self.samp_dir = Path(dcfg["samples_dir"])
        for d in [self.out_dir, self.ckpt_dir, self.samp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _build_data(self):
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]
        dataset = IlluminationDataset(
            normal_dir = dcfg["normal_dir"],
            over_dir   = dcfg["overexposed_dir"],
            under_dir  = dcfg["underexposed_dir"],
            image_size = dcfg["image_size"],
            augment    = dcfg.get("augment", True),
        )
        logger.info(f"Dataset: {len(dataset)} items")
        self.loader = DataLoader(
            dataset,
            batch_size  = tcfg["batch_size"],
            shuffle     = True,
            num_workers = min(4, os.cpu_count() or 1),
            pin_memory  = True,
            collate_fn  = collate_fn,
            drop_last   = True,
        )

    def _build_model(self):
        mcfg = self.cfg["model"]
        dcfg = self.cfg["diffusion"]
        tcfg = self.cfg["training"]

        unet = IlluminationUNetV2(
            image_size            = self.cfg["data"]["image_size"],
            base_channels         = mcfg["base_channels"],
            channel_mult          = tuple(mcfg["channel_mult"]),
            attention_resolutions = tuple(mcfg["attention_resolutions"]),
            num_res_blocks        = mcfg["num_res_blocks"],
            dropout               = mcfg["dropout"],
            num_classes           = mcfg["num_classes"],
        ).to(self.device)

        self.diffusion = GaussianDiffusion(
            model          = unet,
            timesteps      = dcfg["timesteps"],
            beta_schedule  = dcfg["beta_schedule"],
            snr_gamma      = dcfg.get("snr_gamma", 5.0),
            aux_loss_t_max = dcfg.get("aux_loss_t_frac", 0.35),
            device         = self.device,
        ).to(self.device)

        self.ema = EMA(unet, decay=0.9999)

        n = sum(p.numel() for p in unet.parameters()) / 1e6
        logger.info(f"UNet: {n:.2f}M parameters")

    def _build_optim(self):
        tcfg = self.cfg["training"]
        self.optimizer = build_optimizer(self.diffusion.model, lr=tcfg["learning_rate"])
        total_steps    = tcfg["num_epochs"] * len(self.loader)
        self.scheduler = build_scheduler(self.optimizer, tcfg["warmup_steps"], total_steps)
        self.scaler    = GradScaler(enabled=tcfg["mixed_precision"])

    def _build_loss(self):
        tcfg = self.cfg["training"]
        self.criterion = TotalLoss(
            device            = self.device,
            lambda_diffusion  = tcfg["lambda_diffusion"],
            lambda_perceptual = tcfg["lambda_perceptual"],
            lambda_exposure   = tcfg["lambda_exposure"],
            lambda_structure  = tcfg["lambda_structure"],
        )

    def _init_wandb(self):
        self.use_wandb = self.cfg["wandb"]["enabled"]
        if self.use_wandb:
            import wandb
            wandb.init(
                project = self.cfg["wandb"]["project"],
                name    = self.cfg["wandb"]["run_name"],
                config  = self.cfg,
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        tcfg   = self.cfg["training"]
        dcfg   = self.cfg["diffusion"]
        ccfg   = self.cfg["checkpoint"]
        epochs = tcfg["num_epochs"]
        ga     = tcfg["gradient_accumulation_steps"]
        ncls   = self.cfg["model"]["num_classes"]   # null class index = ncls

        for epoch in range(self.start_epoch, epochs):
            self.diffusion.model.train()
            epoch_losses = []
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            self.optimizer.zero_grad()

            for step, batch in enumerate(pbar):
                L_normal  = batch["L_normal"].to(self.device,  non_blocking=True)
                L_target  = batch["L_target"].to(self.device,  non_blocking=True)
                labels    = batch["class_label"].to(self.device, non_blocking=True)
                B         = L_normal.shape[0]

                # CFG: randomly drop class → null
                cfg_mask = torch.rand(B, device=self.device) < tcfg["cfg_dropout_prob"]
                labels_in = labels.clone()
                labels_in[cfg_mask] = ncls

                # Sample timesteps uniformly
                t = torch.randint(0, self.diffusion.T, (B,), device=self.device)

                with autocast(enabled=tcfg["mixed_precision"]):
                    noise_pred, noise, x0_pred, snr_w = self.diffusion.p_losses(
                        x0=L_target, cond=L_normal, c=labels_in, t=t,
                    )

                    # Boolean mask: aux losses valid only where t is low-noise
                    aux_mask = self.diffusion.aux_loss_mask(t)

                    losses = self.criterion(
                        noise_pred   = noise_pred,
                        noise_target = noise,
                        x0_pred      = x0_pred,
                        x0_target    = L_target,
                        normal_L     = L_normal,
                        class_labels = labels,
                        snr_w        = snr_w,
                        aux_mask     = aux_mask,
                    )
                    loss = losses["total"] / ga

                self.scaler.scale(loss).backward()
                epoch_losses.append(losses["total"].item())

                if (step + 1) % ga == 0 or (step + 1) == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), tcfg["grad_clip"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema.update(self.diffusion.model)
                    self.global_step += 1

                pbar.set_postfix({
                    "loss":  f"{losses['total'].item():.4f}",
                    "exp":   f"{losses['exposure']:.3f}",
                    "n_aux": losses["n_aux_samples"],
                })

                if self.global_step % ccfg["log_every_n_steps"] == 0:
                    self._log_step(losses)

            mean_loss = float(np.mean(epoch_losses))
            logger.info(
                f"Epoch {epoch+1:4d}  loss={mean_loss:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )

            # Always save latest
            save_checkpoint(
                path=self.ckpt_dir / "latest.pt",
                model=self.diffusion.model, ema=self.ema,
                optimizer=self.optimizer, scheduler=self.scheduler,
                epoch=epoch, global_step=self.global_step, loss=mean_loss,
            )

            # Save best
            if mean_loss < self.best_metric:
                self.best_metric = mean_loss
                save_checkpoint(
                    path=self.ckpt_dir / "best.pt",
                    model=self.diffusion.model, ema=self.ema,
                    optimizer=self.optimizer, scheduler=self.scheduler,
                    epoch=epoch, global_step=self.global_step, loss=mean_loss,
                )
                logger.info(f"  ✓ Best saved (loss={mean_loss:.4f})")

            if (epoch + 1) % ccfg["save_every_n_epochs"] == 0:
                self._generate_samples(epoch + 1)

        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Sample generation — ALWAYS uses EMA weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, epoch: int):
        """
        FIX: Always swaps EMA weights in before sampling, then restores.
        Previously sometimes used live training weights → noisy samples.
        """
        self.diffusion.model.eval()
        dcfg = self.cfg["diffusion"]
        ncls = self.cfg["model"]["num_classes"]

        batch    = next(iter(self.loader))
        L_normal = batch["L_normal"][:1].to(self.device)
        AB_normal = batch["AB_normal"][:1].to(self.device)

        # ── Swap to EMA for clean inference ──────────────────────────
        self.ema.apply_shadow(self.diffusion.model)

        saved_paths = []
        for cls_id, cls_name in [(0, "over"), (1, "under")]:
            c     = torch.tensor([cls_id], device=self.device)
            L_gen = self.diffusion.ddim_sample(
                cond           = L_normal,
                c              = c,
                ddim_steps     = dcfg["ddim_steps"],
                eta            = dcfg["ddim_eta"],
                guidance_scale = dcfg["classifier_free_guidance_scale"],
                null_class_idx = ncls,
            )
            img  = self._lab_to_pil(L_gen, AB_normal)
            path = self.samp_dir / f"epoch{epoch:04d}_{cls_name}.png"
            img.save(str(path))
            saved_paths.append(str(path))

        # Normal reference
        img_n  = self._lab_to_pil(L_normal, AB_normal)
        path_n = self.samp_dir / f"epoch{epoch:04d}_normal.png"
        img_n.save(str(path_n))

        # ── Restore live weights ──────────────────────────────────────
        self.ema.restore(self.diffusion.model)
        self.diffusion.model.train()

        logger.info(f"  Samples saved → {self.samp_dir}")

        if self.use_wandb:
            import wandb
            wandb.log({
                "samples/over":   wandb.Image(saved_paths[0]),
                "samples/under":  wandb.Image(saved_paths[1]),
                "samples/normal": wandb.Image(str(path_n)),
                "epoch": epoch,
            })

    @staticmethod
    def _lab_to_pil(L_tensor: torch.Tensor, AB_tensor: torch.Tensor) -> Image.Image:
        L_np  = L_tensor[0, 0].cpu().float().numpy()
        AB_np = AB_tensor[0].cpu().float().numpy().transpose(1, 2, 0)
        L_r   = denormalise_L(L_np)
        AB_r  = denormalise_AB(AB_np)
        lab   = np.concatenate([L_r[:, :, None], AB_r], axis=2)
        return Image.fromarray(lab_to_rgb(lab))

    def _log_step(self, losses: dict):
        if self.use_wandb:
            import wandb
            wandb.log({f"train/{k}": v for k, v in losses.items()
                       if k != "n_aux_samples"} | {"step": self.global_step})
        else:
            logger.debug(
                f"step={self.global_step}  " +
                "  ".join(f"{k}={v:.4f}" for k, v in losses.items()
                          if k not in ("total", "n_aux_samples")) +
                f"  total={losses['total'].item():.4f}"
            )

    # ------------------------------------------------------------------
    # Generate full synthetic dataset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_synthetic_dataset(
        self,
        normal_dir:     str,
        output_dir:     str,
        ddim_steps:     int   = 50,
        guidance_scale: float = 4.0,
        checkpoint:     str   = "best",
    ):
        ckpt_path = self.ckpt_dir / f"{checkpoint}.pt"
        load_checkpoint(ckpt_path, self.diffusion.model, self.ema, device=self.device)
        self.ema.apply_shadow(self.diffusion.model)
        self.diffusion.model.eval()

        dcfg = self.cfg["diffusion"]
        ncls = self.cfg["model"]["num_classes"]
        sz   = self.cfg["data"]["image_size"]

        out_over  = Path(output_dir) / "over"
        out_under = Path(output_dir) / "under"
        out_over.mkdir(parents=True, exist_ok=True)
        out_under.mkdir(parents=True, exist_ok=True)

        normal_paths = IlluminationDataset._collect(normal_dir)
        logger.info(f"Generating pairs for {len(normal_paths)} normal images…")

        from dataset.illumination_dataset import rgb_to_lab, normalise_L, normalise_AB
        from tqdm import tqdm

        for img_path in tqdm(normal_paths, desc="Generating"):
            img    = Image.open(img_path).convert("RGB").resize((sz, sz), Image.BICUBIC)
            img_np = np.array(img)
            lab    = rgb_to_lab(img_np)
            L_n    = normalise_L(lab[:, :, 0])
            AB_n   = normalise_AB(lab[:, :, 1:])

            L_t  = torch.from_numpy(L_n).unsqueeze(0).unsqueeze(0).to(self.device)
            AB_t = torch.from_numpy(AB_n.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            stem = Path(img_path).stem

            for cls_id, out_dir in [(0, out_over), (1, out_under)]:
                c     = torch.tensor([cls_id], device=self.device)
                L_gen = self.diffusion.ddim_sample(
                    cond=L_t, c=c, ddim_steps=ddim_steps, eta=0.0,
                    guidance_scale=guidance_scale, null_class_idx=ncls,
                )
                out_img = self._lab_to_pil(L_gen, AB_t)
                out_img.save(str(out_dir / f"{stem}.png"))

        self.ema.restore(self.diffusion.model)
        logger.info(f"Done → {output_dir}")
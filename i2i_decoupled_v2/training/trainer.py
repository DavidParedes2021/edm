"""
training/trainer.py
--------------------
Main training loop for the illumination diffusion pipeline.

Key engineering decisions
--------------------------
* fp16 mixed precision via torch.cuda.amp (GradScaler + autocast).
* Gradient accumulation to decouple batch-size from VRAM.
* Classifier-Free Guidance training: 15% of steps replace c with null label.
* EMA (Exponential Moving Average) of model weights for stable inference.
* Saves only (a) the best checkpoint and (b) the latest checkpoint.
* Periodic sample generation to visually monitor progress.
* Optional W&B integration.
"""

import os
import copy
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset.illumination_dataset import (
    IlluminationDataset,
    collate_fn,
    denormalise_L,
    denormalise_AB,
    lab_to_rgb,
)
from model.unet import IlluminationUNetV2
from model.diffusion import GaussianDiffusion
from training.losses import TotalLoss
from utils.metrics import evaluate_sample
from utils.misc import (
    set_seed,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    EMA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

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
    # Setup helpers
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
            normal_dir   = dcfg["normal_dir"],
            over_dir     = dcfg["overexposed_dir"],
            under_dir    = dcfg["underexposed_dir"],
            image_size   = dcfg["image_size"],
            augment      = dcfg.get("augment", True),
        )
        logger.info(f"Dataset: {len(dataset)} items")

        self.loader = DataLoader(
            dataset,
            batch_size     = tcfg["batch_size"],
            shuffle        = True,
            num_workers    = min(4, os.cpu_count() or 1),
            pin_memory     = True,
            collate_fn     = collate_fn,
            drop_last      = True,
        )

    def _build_model(self):
        mcfg = self.cfg["model"]
        dcfg = self.cfg["diffusion"]

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
            model         = unet,
            timesteps     = dcfg["timesteps"],
            beta_schedule = dcfg["beta_schedule"],
            device        = self.device,
        ).to(self.device)

        # EMA model for inference
        self.ema = EMA(unet, decay=0.9999)

        total_params = sum(p.numel() for p in unet.parameters()) / 1e6
        logger.info(f"UNet parameters: {total_params:.2f}M")

    def _build_optim(self):
        tcfg = self.cfg["training"]
        self.optimizer = build_optimizer(
            self.diffusion.model,
            lr=tcfg["learning_rate"],
        )
        steps_per_epoch = len(self.loader)
        total_steps     = tcfg["num_epochs"] * steps_per_epoch
        self.scheduler  = build_scheduler(self.optimizer, tcfg["warmup_steps"], total_steps)
        self.scaler     = GradScaler(enabled=tcfg["mixed_precision"])

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
    # Training
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
                L_normal = batch["L_normal"].to(self.device, non_blocking=True)
                L_target = batch["L_target"].to(self.device, non_blocking=True)
                AB_normal = batch["AB_normal"].to(self.device, non_blocking=True)
                labels   = batch["class_label"].to(self.device, non_blocking=True)

                B = L_normal.shape[0]

                # ── CFG: randomly replace labels with null ──────────
                cfg_mask = torch.rand(B, device=self.device) < tcfg["cfg_dropout_prob"]
                labels_cfg = labels.clone()
                labels_cfg[cfg_mask] = ncls   # null class

                # ── Sample timesteps ───────────────────────────────
                t = torch.randint(0, self.diffusion.T, (B,), device=self.device)

                # ── Forward + loss ─────────────────────────────────
                with autocast(enabled=tcfg["mixed_precision"]):
                    noise_pred, noise = self.diffusion.p_losses(
                        x0   = L_target,
                        cond = L_normal,
                        c    = labels_cfg,
                        t    = t,
                    )

                    # Reconstruct x0 from noise prediction for auxiliary losses
                    sq_a  = self.diffusion._extract(
                        self.diffusion.sqrt_alpha_cumprod, t, L_target.shape
                    )
                    sq_1a = self.diffusion._extract(
                        self.diffusion.sqrt_one_minus_alpha_cumprod, t, L_target.shape
                    )
                    # x_t = sq_a * x0 + sq_1a * eps  →  x0_pred = (x_t - sq_1a * eps_pred) / sq_a
                    x_t    = sq_a * L_target + sq_1a * noise
                    x0_pred = (x_t - sq_1a * noise_pred) / (sq_a + 1e-8)
                    x0_pred = x0_pred.clamp(-1.0, 1.0)

                    losses = self.criterion(
                        noise_pred   = noise_pred,
                        noise_target = noise,
                        x0_pred      = x0_pred,
                        x0_target    = L_target,
                        normal_L     = L_normal,
                        class_labels = labels,
                    )
                    loss = losses["total"] / ga

                self.scaler.scale(loss).backward()
                epoch_losses.append(losses["total"].item())

                # ── Gradient accumulation step ──────────────────────
                if (step + 1) % ga == 0 or (step + 1) == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.diffusion.model.parameters(),
                        tcfg["grad_clip"],
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema.update(self.diffusion.model)
                    self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "exp":  f"{losses['exposure']:.4f}",
                    "perc": f"{losses['perceptual']:.4f}",
                })

                if self.global_step % ccfg["log_every_n_steps"] == 0:
                    self._log_step(losses)

            # ── Epoch end ────────────────────────────────────────────
            mean_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1}  loss={mean_loss:.4f}")

            # Save latest always
            save_checkpoint(
                path       = self.ckpt_dir / "latest.pt",
                model      = self.diffusion.model,
                ema        = self.ema,
                optimizer  = self.optimizer,
                scheduler  = self.scheduler,
                epoch      = epoch,
                global_step= self.global_step,
                loss       = mean_loss,
            )

            # Save best
            if mean_loss < self.best_metric:
                self.best_metric = mean_loss
                save_checkpoint(
                    path       = self.ckpt_dir / "best.pt",
                    model      = self.diffusion.model,
                    ema        = self.ema,
                    optimizer  = self.optimizer,
                    scheduler  = self.scheduler,
                    epoch      = epoch,
                    global_step= self.global_step,
                    loss       = mean_loss,
                )
                logger.info(f"  ✓ Best checkpoint saved (loss={mean_loss:.4f})")

            # Periodic sample generation
            if (epoch + 1) % ccfg["save_every_n_epochs"] == 0:
                self._generate_samples(epoch + 1)

        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Inference / sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, epoch: int):
        """Generate one over- and one under-exposed sample from the first batch."""
        self.diffusion.model.eval()
        dcfg   = self.cfg["diffusion"]
        ncls   = self.cfg["model"]["num_classes"]

        # Grab one normal image from the loader
        batch = next(iter(self.loader))
        L_normal = batch["L_normal"][:1].to(self.device)   # [1,1,H,W]
        AB_normal = batch["AB_normal"][:1].to(self.device) # [1,2,H,W]

        # Swap EMA weights in for inference
        self.ema.apply_shadow(self.diffusion.model)

        sample_paths = []
        for cls_id, cls_name in [(0, "over"), (1, "under")]:
            c = torch.tensor([cls_id], device=self.device)

            L_gen = self.diffusion.ddim_sample(
                cond            = L_normal,
                c               = c,
                ddim_steps      = dcfg["ddim_steps"],
                eta             = dcfg["ddim_eta"],
                guidance_scale  = dcfg["classifier_free_guidance_scale"],
                null_class_idx  = ncls,
            )   # [1,1,H,W]

            img = self._lab_tensor_to_pil(L_gen, AB_normal)
            path = self.samp_dir / f"epoch{epoch:04d}_{cls_name}.png"
            img.save(str(path))
            sample_paths.append(str(path))

        # Also save normal for reference
        img_n = self._lab_tensor_to_pil(L_normal, AB_normal)
        path_n = self.samp_dir / f"epoch{epoch:04d}_normal.png"
        img_n.save(str(path_n))

        # Restore original weights
        self.ema.restore(self.diffusion.model)
        self.diffusion.model.train()

        logger.info(f"  Samples saved → {self.samp_dir}")

        if self.use_wandb:
            import wandb
            wandb.log({
                "samples/over":   wandb.Image(sample_paths[0]),
                "samples/under":  wandb.Image(sample_paths[1]),
                "samples/normal": wandb.Image(str(path_n)),
                "epoch": epoch,
            })

    @staticmethod
    def _lab_tensor_to_pil(L_tensor: torch.Tensor, AB_tensor: torch.Tensor) -> Image.Image:
        """Convert L [1,1,H,W] + AB [1,2,H,W] tensors → PIL RGB image."""
        L_np  = L_tensor[0, 0].cpu().float().numpy()    # HxW
        AB_np = AB_tensor[0].cpu().float().numpy()      # 2xHxW

        # Denormalise
        L_real  = denormalise_L(L_np)                   # [0,100]
        AB_real = denormalise_AB(AB_np.transpose(1, 2, 0))  # HxWx2

        lab = np.concatenate(
            [L_real[:, :, np.newaxis], AB_real], axis=2  # HxWx3
        )
        rgb = lab_to_rgb(lab)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, losses: dict):
        if self.use_wandb:
            import wandb
            wandb.log({
                "train/loss_total":      losses["total"].item(),
                "train/loss_diffusion":  losses["diffusion"],
                "train/loss_perceptual": losses["perceptual"],
                "train/loss_exposure":   losses["exposure"],
                "train/loss_structure":  losses["structure"],
                "step": self.global_step,
            })
        else:
            logger.debug(
                f"step={self.global_step}  "
                f"total={losses['total'].item():.4f}  "
                f"diff={losses['diffusion']:.4f}  "
                f"perc={losses['perceptual']:.4f}  "
                f"exp={losses['exposure']:.4f}  "
                f"struc={losses['structure']:.4f}"
            )

    # ------------------------------------------------------------------
    # Public: generate full synthetic dataset
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
        """
        Generate a full synthetic (over, under) paired dataset from all normal
        images in normal_dir.  Saves results under output_dir/{over,under}/.
        """
        from dataset.illumination_dataset import IlluminationDataset, collate_fn
        import torchvision.transforms as T

        ckpt_path = self.ckpt_dir / f"{checkpoint}.pt"
        load_checkpoint(ckpt_path, self.diffusion.model, self.ema)
        self.ema.apply_shadow(self.diffusion.model)
        self.diffusion.model.eval()

        dcfg = self.cfg["diffusion"]
        ncls = self.cfg["model"]["num_classes"]
        sz   = self.cfg["data"]["image_size"]

        out_over  = Path(output_dir) / "over"
        out_under = Path(output_dir) / "under"
        out_over.mkdir(parents=True, exist_ok=True)
        out_under.mkdir(parents=True, exist_ok=True)

        transform = T.Compose([T.Resize((sz, sz)), T.ToTensor()])

        normal_paths = IlluminationDataset._collect(normal_dir)
        logger.info(f"Generating synthetic pairs for {len(normal_paths)} normal images…")

        for img_path in tqdm(normal_paths, desc="Generating"):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((sz, sz), Image.BICUBIC)
            img_np = np.array(img)

            from dataset.illumination_dataset import rgb_to_lab, normalise_L, normalise_AB, denormalise_L, denormalise_AB

            from skimage.color import rgb2lab
            lab = rgb_to_lab(img_np)
            L_n  = normalise_L(lab[:, :, 0])
            AB_n = normalise_AB(lab[:, :, 1:])

            L_tensor  = torch.from_numpy(L_n).unsqueeze(0).unsqueeze(0).to(self.device)   # [1,1,H,W]
            AB_tensor = torch.from_numpy(AB_n.transpose(2, 0, 1)).unsqueeze(0).to(self.device)  # [1,2,H,W]

            stem = Path(img_path).stem

            for cls_id, cls_name, out_dir in [(0, "over", out_over), (1, "under", out_under)]:
                c = torch.tensor([cls_id], device=self.device)
                L_gen = self.diffusion.ddim_sample(
                    cond           = L_tensor,
                    c              = c,
                    ddim_steps     = ddim_steps,
                    eta            = 0.0,
                    guidance_scale = guidance_scale,
                    null_class_idx = ncls,
                )
                out_img = self._lab_tensor_to_pil(L_gen, AB_tensor)
                out_img.save(str(out_dir / f"{stem}.png"))

        self.ema.restore(self.diffusion.model)
        logger.info(f"Done. Pairs saved to {output_dir}")

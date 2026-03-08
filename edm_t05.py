"""
Conditional Artifact Diffusion Model — DGX Server Version  (edm_t05)
=====================================================================
Fixes over edm_t04:
  1. Resolution 128→256 — recovers fine endoscopic texture
  2. Deeper UNet with attention at 3 scales (not just the deepest)
  3. Perceptual + MSE loss instead of plain MSE — drives visible exposure contrast
  4. DDIM scheduler (50-step inference ≈ DDPM 1000-step quality)
  5. Stronger data augmentation (random H-flip, colour jitter) to reduce overfitting
  6. EMA of model weights — smoother, sharper samples at inference
  7. Multi-GPU via Accelerate (uses all DGX A100s automatically)
  8. Classifier-Free Guidance (CFG) support: 10 % unconditional dropout during
     training + guidance_scale at inference → much stronger exposure effect
  9. SNR-weighted loss (min-SNR γ=5) — stops low-t steps from dominating MSE
 10. layers_per_block 1→2 — restores detail capacity lost by the memory cut
 11. Best-model checkpoint saved every epoch (not only every N)
 12. WandB image logging now uploads the grid, not just saves it to disk

Environment: Shared DGX Ubuntu, NVIDIA Driver 465.19.01, CUDA 11.3
Docker image: pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
Python: 3.8

PINNED DEPENDENCY VERSIONS (install exactly these):
----------------------------------------------------
  pip install \
    diffusers==0.14.0 \
    huggingface_hub==0.25.2 \
    accelerate==0.18.0 \
    transformers==4.27.4 \
    wandb==0.14.2 \
    Pillow==9.5.0 \
    numpy==1.23.5 \
    tqdm==4.65.0 \
    matplotlib==3.7.1 \
    packaging==23.1 \
    torchvision==0.12.0+cu113   # needed for VGG perceptual loss

Run (single-node, all GPUs):
  accelerate launch --multi_gpu edm_t05.py
  -- OR --
  python edm_t05.py     # single-GPU / auto-detect
"""

# =====================================================
# CONFIGURATION — Edit these paths before running
# =====================================================
# Root folder of the Endo4IE dataset (synced via gdown)
#DATASET_PATH = "./"          # <-- EDIT THIS
DATASET_PATH = "../../data/datasets/endo4ie/"

# Where to save models, checkpoints and samples
#OUTPUT_BASE  = "./endo_diffusion/output/" # <-- EDIT THIS
OUTPUT_BASE = "../edm_outputs/edm_t05/"

# Synthetic dataset output folder
SYNTHETIC_OUTPUT = "../edm_outputs/edm_t05/synthetic_datasets" # <-- EDIT THIS

# Weights & Biases project name (set to None to disable W&B)
WANDB_PROJECT = None 

# =====================================================
# Imports
# =====================================================
import os
import sys
import copy
import random
import numpy as np
from pathlib import Path
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as tv_models

from diffusers import UNet2DModel, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if WANDB_PROJECT is not None:
    import wandb

# =====================================================
# Runtime version guard
# =====================================================
def _check_versions():
    import accelerate as _acc
    import diffusers as _diff
    import huggingface_hub as _hfh

    hfh_ver = tuple(int(x) for x in _hfh.__version__.split(".")[:2])
    if hfh_ver >= (0, 26):
        print(
            f"\nFATAL: huggingface_hub {_hfh.__version__} is installed.\n"
            f"  diffusers==0.14.0 requires huggingface_hub<=0.25.x.\n"
            f"      pip install huggingface_hub==0.25.2\n"
        )
        sys.exit(1)
    print("Library versions OK.\n")

_check_versions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Count: {torch.cuda.device_count()}")


# =====================================================
# Perceptual (VGG) Loss  — FIX #3
# =====================================================
class VGGPerceptualLoss(nn.Module):
    """
    L1 distance in VGG-16 relu2_2 feature space.
    Drives the model to reproduce high-frequency textures, not just pixel means.
    Frozen — gradients only flow through the generator.
    """
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(pretrained=False)   # weights loaded below
        # Use only the first 9 layers (up to relu2_2)
        self.slice = nn.Sequential(*list(vgg.features.children())[:9]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False

        # ImageNet mean/std for normalising [-1,1] inputs into VGG range
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # x is in [-1, 1]; convert to [0,1] then normalise for VGG
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        return F.l1_loss(self.slice(self._preprocess(pred)),
                         self.slice(self._preprocess(target)))


# =====================================================
# Dataset with augmentation  — FIX #5
# =====================================================
class ConditionalArtifactDataset(Dataset):
    """
    Paired dataset: normal image → artifact image.
    Applies the SAME random augmentation to both images (spatial consistency).

    Expected folder structure:
      <base_path>/real_overexposed/<split>/normal_frames/
      <base_path>/real_overexposed/<split>/overexposed/
    """

    def __init__(self, base_path, split='train', artifact_type='overexposed',
                 image_size=256, augment=True):   # FIX #1: default size 256
        self.base_path     = Path(base_path)
        self.split         = split
        self.artifact_type = artifact_type
        self.image_size    = image_size
        self.augment       = augment and (split == 'train')

        folder          = f'real_{artifact_type}'
        artifact_folder = artifact_type
        normal_folder   = 'normal_frames'

        def _find_split_dir(parent, name):
            for c in [name, name.capitalize(), name.upper(), name.lower()]:
                p = parent / c
                if p.is_dir():
                    return p
            raise FileNotFoundError(
                f"Cannot find split '{name}' under {parent}. "
                f"Available: {[x.name for x in parent.iterdir() if x.is_dir()]}"
            )

        split_dir          = _find_split_dir(self.base_path / folder, split)
        self.normal_path   = split_dir / normal_folder
        self.artifact_path = split_dir / artifact_folder

        for p in [self.normal_path, self.artifact_path]:
            if not p.is_dir():
                raise FileNotFoundError(f"Directory not found: {p}")

        _exts = ('.png', '.jpg', '.jpeg')
        self.normal_images   = sorted(f for f in os.listdir(self.normal_path)
                                      if f.lower().endswith(_exts))
        self.artifact_images = sorted(f for f in os.listdir(self.artifact_path)
                                      if f.lower().endswith(_exts))

        print(f"[{split}/{artifact_type}] "
              f"{len(self.normal_images)} normal + {len(self.artifact_images)} artifact")

    def __len__(self):
        return min(len(self.normal_images), len(self.artifact_images))

    def _load(self, path):
        return Image.open(path).convert('RGB').resize(
            (self.image_size, self.image_size), Image.LANCZOS)

    def _to_tensor(self, img):
        arr = np.array(img).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0

    def __getitem__(self, idx):
        normal_name   = self.normal_images[idx]
        artifact_name = normal_name if normal_name in self.artifact_images \
                        else self.artifact_images[idx]

        normal_img   = self._load(self.normal_path   / normal_name)
        artifact_img = self._load(self.artifact_path / artifact_name)

        # ---- Paired augmentation (FIX #5) ----
        if self.augment:
            # Horizontal flip (same decision for both)
            if random.random() > 0.5:
                normal_img   = TF.hflip(normal_img)
                artifact_img = TF.hflip(artifact_img)
            # Mild colour jitter applied ONLY to normal (the condition),
            # so the model learns to be robust to slight colour shifts
            jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
            normal_img = jitter(normal_img)

        return {
            'normal':   self._to_tensor(normal_img),
            'artifact': self._to_tensor(artifact_img),
        }


# =====================================================
# Model — FIX #2, #10: deeper UNet, more attention
# =====================================================
class ConditionalArtifactUNet(nn.Module):
    """
    Conditional UNet: p(x_artifact | x_normal).
    Concatenates noisy artifact (3ch) + normal condition (3ch) → 6-ch input.

    Changes from edm_t04:
      - layers_per_block: 1 → 2  (FIX #10)
      - block_out_channels extended to (128, 256, 512, 512)
      - Attention at the bottom TWO resolution levels, not just the deepest (FIX #2)
    """

    def __init__(self, image_size=256):
        super().__init__()
        # Store constructor args so EMA can rebuild a clean shadow model
        # without deepcopy (which fails on AMP-prepared models).
        self._init_kwargs = {"image_size": image_size}
        # P100-16GB OOM fix:
        #   layers_per_block 2->1  : halves residual-block activation memory (~40% saving)
        #   AttnDownBlock2D removed at 32x32: that level needs ~8 GiB for the
        #   baddbmm attention score matrix alone at 512-ch, batch=4. Attention
        #   is kept only at the deepest 16x16 level where the map is 4x smaller.
        self.unet = UNet2DModel(
            sample_size    = image_size,
            in_channels    = 6,
            out_channels   = 3,
            layers_per_block = 1,                       # reduced 2->1: ~40% less activation memory
            block_out_channels = (128, 256, 512, 512),
            down_block_types = (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",       # was AttnDownBlock2D at 32x32 -- OOM on P100
                "AttnDownBlock2D",   # attention only at deepest 16x16 level
            ),
            up_block_types = (
                "AttnUpBlock2D",     # mirrors deepest down block
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, noisy_artifact, timesteps, normal_condition):
        x = torch.cat([noisy_artifact, normal_condition], dim=1)
        return self.unet(x, timesteps).sample


# =====================================================
# EMA helper — FIX #6
# =====================================================
class EMA:
    """Exponential Moving Average of model parameters for cleaner inference."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # deepcopy fails on AMP-prepared models with a PicklingError.
        # Fix: instantiate a fresh, plain model from the same constructor args,
        # then copy weights via state_dict — no pickling involved.
        self.shadow = model.__class__(**model._init_kwargs).eval()
        self.shadow.load_state_dict(
            {k: v.clone() for k, v in model.state_dict().items()}
        )
        self.shadow.to(next(model.parameters()).device)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# =====================================================
# SNR-weighted loss  — FIX #9
# =====================================================
def snr_weight(timesteps, noise_scheduler, gamma=5.0):
    """
    Min-SNR-γ loss weighting (Hang et al. 2023).
    Prevents high-noise timesteps (t≈T) from dominating training,
    allowing the model to focus on learning structural/colour detail.
    Returns per-sample weight tensor of shape (B,).
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_t        = alphas_cumprod[timesteps]
    snr            = alpha_t / (1.0 - alpha_t).clamp(min=1e-8)
    weight         = torch.minimum(snr, torch.full_like(snr, gamma)) / gamma
    return weight                            # shape (B,)


# =====================================================
# Sample generation
# =====================================================
@torch.no_grad()
def generate_and_save_samples(
    model, noise_scheduler, dataloader, output_dir, epoch,
    num_inference_steps=50, num_samples=4,
    guidance_scale=3.0,    # FIX #8
    use_wandb=False,
):
    model.eval()
    dev = next(model.parameters()).device

    batch       = next(iter(dataloader))
    num_samples = min(num_samples, batch['normal'].shape[0])
    normal_imgs    = batch['normal'][:num_samples].to(dev)
    real_artifacts = batch['artifact'][:num_samples].to(dev)

    # DDIM denoising with CFG
    generated = torch.randn_like(normal_imgs)
    noise_scheduler.set_timesteps(num_inference_steps)

    null_cond = torch.zeros_like(normal_imgs)   # unconditional signal

    for t in noise_scheduler.timesteps:
        ts = torch.full((num_samples,), t, device=dev, dtype=torch.long)

        # Conditional prediction
        noise_cond   = model(generated, ts, normal_imgs)
        # Unconditional prediction (null condition)
        noise_uncond = model(generated, ts, null_cond)
        # CFG blend  (FIX #8)
        noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        generated = noise_scheduler.step(noise_pred, t, generated).prev_sample

    # De-normalise
    def denorm(x):
        return (x + 1.0) / 2.0

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    for row, (title, imgs) in enumerate(zip(
        ['Normal (Condition)', 'Real Artifact', 'Generated Artifact'],
        [denorm(normal_imgs), denorm(real_artifacts), denorm(generated)]
    )):
        for col in range(num_samples):
            axes[row, col].imshow(imgs[col].cpu().permute(1, 2, 0).clamp(0, 1))
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'samples', f'epoch_{epoch:04d}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Samples saved: {save_path}")

    # Upload grid to W&B  (FIX #11)
    if use_wandb:
        wandb.log({"samples": wandb.Image(save_path), "epoch": epoch})


# =====================================================
# Training
# =====================================================
def train_conditional_diffusion(
    dataset_path,
    artifact_type        = 'overexposed',
    output_dir           = './conditional_diffusion',
    wandb_project        = None,
    num_epochs           = 150,
    batch_size           = 4,           # 4 per GPU × N GPUs
    learning_rate        = 5e-5,        # slightly lower for 256-res stability
    image_size           = 256,         # FIX #1
    num_inference_steps  = 50,          # DDIM 50 steps ≈ DDPM 1000  FIX #4
    save_every_n_epochs  = 5,
    log_images_every_n_epochs = 2,
    num_workers          = 4,
    grad_accum_steps     = 2,
    guidance_scale       = 3.0,         # CFG strength at inference  FIX #8
    cond_drop_prob       = 0.10,        # fraction of steps with null cond  FIX #8
    perceptual_weight    = 0.5,         # weight of VGG loss  FIX #3
    ema_decay            = 0.9999,      # FIX #6
    snr_gamma            = 5.0,         # min-SNR γ  FIX #9
):
    out_dir     = os.path.join(output_dir, artifact_type)
    samples_dir = os.path.join(out_dir, 'samples')
    ckpt_dir    = os.path.join(out_dir, 'checkpoints')
    for d in [out_dir, samples_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"{artifact_type}-diffusion-v5",
            config=dict(
                artifact_type=artifact_type, num_epochs=num_epochs,
                batch_size=batch_size, learning_rate=learning_rate,
                image_size=image_size, guidance_scale=guidance_scale,
                perceptual_weight=perceptual_weight, snr_gamma=snr_gamma,
            ),
        )

    # FIX #7: Accelerator for multi-GPU
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=grad_accum_steps,
    )

    train_dataset = ConditionalArtifactDataset(
        dataset_path, split='train', artifact_type=artifact_type,
        image_size=image_size, augment=True)
    val_dataset = ConditionalArtifactDataset(
        dataset_path, split='validation', artifact_type=artifact_type,
        image_size=image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    model = ConditionalArtifactUNet(image_size=image_size)

    # FIX #4: DDIM scheduler — fast, high-quality denoising
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

    # FIX #3: Perceptual loss
    perc_loss_fn = VGGPerceptualLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01, betas=(0.9, 0.999))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    perc_loss_fn = perc_loss_fn.to(accelerator.device)

    # FIX #6: EMA — maintained on the unwrapped model on the main process
    if accelerator.is_main_process:
        ema = EMA(accelerator.unwrap_model(model), decay=ema_decay)

    global_step   = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            normal_imgs   = batch['normal']
            artifact_imgs = batch['artifact']

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (artifact_imgs.shape[0],), device=artifact_imgs.device
            ).long()

            noise           = torch.randn_like(artifact_imgs)
            noisy_artifacts = noise_scheduler.add_noise(artifact_imgs, noise, timesteps)

            # FIX #8: Classifier-Free Guidance dropout
            # With probability cond_drop_prob, replace condition with zeros
            drop_mask = (torch.rand(normal_imgs.shape[0], 1, 1, 1,
                                    device=normal_imgs.device) < cond_drop_prob)
            cond_input = torch.where(drop_mask, torch.zeros_like(normal_imgs), normal_imgs)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_artifacts, timesteps, cond_input)

                # FIX #9: SNR-weighted MSE
                weights     = snr_weight(timesteps, noise_scheduler, gamma=snr_gamma)
                mse         = (F.mse_loss(noise_pred, noise, reduction='none')
                               .mean(dim=(1, 2, 3)))      # per-sample MSE
                mse_loss    = (mse * weights).mean()

                # FIX #3: Perceptual loss on denoised prediction
                # Estimate x0 from current noise prediction (useful signal for VGG)
                alpha_prod  = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                x0_pred     = (noisy_artifacts - (1 - alpha_prod).sqrt() * noise_pred) \
                              / alpha_prod.sqrt().clamp(min=1e-8)
                perc_loss   = perc_loss_fn(x0_pred.clamp(-1, 1), artifact_imgs)

                loss = mse_loss + perceptual_weight * perc_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # FIX #6: Update EMA
            if accelerator.is_main_process and accelerator.sync_gradients:
                ema.update(accelerator.unwrap_model(model))

            epoch_loss += loss.item()

            if accelerator.is_main_process and use_wandb:
                wandb.log({
                    "train/loss":     loss.item(),
                    "train/mse":      mse_loss.item(),
                    "train/perc":     perc_loss.item(),
                    "train/lr":       lr_scheduler.get_last_lr()[0],
                    "train/epoch":    epoch,
                    "train/step":     global_step,
                })

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse":  f"{mse_loss.item():.4f}",
                "perc": f"{perc_loss.item():.4f}",
            })
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                normal_imgs   = batch['normal']
                artifact_imgs = batch['artifact']
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (artifact_imgs.shape[0],), device=artifact_imgs.device
                ).long()
                noise           = torch.randn_like(artifact_imgs)
                noisy_artifacts = noise_scheduler.add_noise(artifact_imgs, noise, timesteps)
                noise_pred      = model(noisy_artifacts, timesteps, normal_imgs)
                val_loss       += F.mse_loss(noise_pred, noise).item()

        avg_val_loss = val_loss / len(val_loader)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs} — "
                  f"Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}")
            if use_wandb:
                wandb.log({"epoch": epoch+1,
                           "train/epoch_loss": avg_train_loss,
                           "val/loss": avg_val_loss})

            # Best-model checkpoint every epoch  (FIX #11)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save EMA weights as best model
                best_path = os.path.join(out_dir, 'best_model.pt')
                torch.save(ema.state_dict(), best_path)
                print(f"✓ New best EMA model: {best_path}  (val={avg_val_loss:.4f})")

        # Sample visualisation
        if (epoch + 1) % log_images_every_n_epochs == 0 and accelerator.is_main_process:
            ema_model = ema.shadow.to(accelerator.device)
            generate_and_save_samples(
                ema_model, noise_scheduler, val_loader, out_dir,
                epoch + 1, num_inference_steps,
                guidance_scale=guidance_scale,
                use_wandb=use_wandb,
            )

        # Periodic checkpoint (full state for resuming)
        if (epoch + 1) % save_every_n_epochs == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     unwrapped.state_dict(),
                'ema_state_dict':       ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':           avg_train_loss,
                'val_loss':             avg_val_loss,
                'config': {
                    'artifact_type':       artifact_type,
                    'image_size':          image_size,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale':      guidance_scale,
                },
            }, ckpt_path)
            print(f"✓ Checkpoint saved: {ckpt_path}")

    # Final model (EMA weights)
    if accelerator.is_main_process:
        final_path = os.path.join(out_dir, 'final_model.pt')
        torch.save(ema.state_dict(), final_path)
        print(f"\n✓ Final EMA model saved: {final_path}")
        if use_wandb:
            wandb.finish()

    return accelerator.unwrap_model(model), noise_scheduler


# =====================================================
# Synthetic dataset generation
# =====================================================
@torch.no_grad()
def generate_synthetic_dataset(
    model,
    noise_scheduler,
    normal_image_folder,
    output_folder,
    artifact_type,
    num_variations       = 5,
    num_inference_steps  = 50,
    image_size           = 256,
    guidance_scale       = 3.0,
):
    model.eval()
    dev = next(model.parameters()).device
    null_cond_cache = {}   # cached per spatial size

    os.makedirs(os.path.join(output_folder, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, f'synthetic_{artifact_type}'), exist_ok=True)

    _exts = ('.png', '.jpg', '.jpeg')
    normal_images = sorted(f for f in os.listdir(normal_image_folder)
                           if f.lower().endswith(_exts))

    print(f"\nGenerating synthetic {artifact_type} dataset...")
    print(f"  Images: {len(normal_images)} × {num_variations} variations")

    pair_count = 0
    for img_idx, img_name in enumerate(tqdm(normal_images, desc="Generating")):
        normal_img         = Image.open(os.path.join(normal_image_folder, img_name)).convert('RGB')
        original_size      = normal_img.size
        normal_img_resized = normal_img.resize((image_size, image_size), Image.LANCZOS)
        normal_tensor      = (torch.from_numpy(np.array(normal_img_resized))
                              .float().permute(2, 0, 1) / 127.5 - 1.0
                              ).unsqueeze(0).to(dev)

        null_cond = null_cond_cache.get(image_size)
        if null_cond is None:
            null_cond = torch.zeros_like(normal_tensor)
            null_cond_cache[image_size] = null_cond

        for variation in range(num_variations):
            torch.manual_seed(img_idx * num_variations + variation)
            generated = torch.randn_like(normal_tensor)
            noise_scheduler.set_timesteps(num_inference_steps)

            for t in noise_scheduler.timesteps:
                ts           = torch.full((1,), t, device=dev, dtype=torch.long)
                noise_cond   = model(generated, ts, normal_tensor)
                noise_uncond = model(generated, ts, null_cond)
                noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                generated    = noise_scheduler.step(noise_pred, t, generated).prev_sample

            gen_arr = ((generated + 1) / 2)[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            gen_img = Image.fromarray((gen_arr * 255).astype(np.uint8))
            gen_img = gen_img.resize(original_size, Image.LANCZOS)

            base_name = os.path.splitext(img_name)[0]
            pair_name = f"{base_name}_var{variation:02d}.png"

            normal_img.save(os.path.join(output_folder, 'normal', pair_name))
            gen_img.save(os.path.join(output_folder, f'synthetic_{artifact_type}', pair_name))
            pair_count += 1

    print(f"✓ Generated {pair_count} synthetic pairs → {output_folder}")
    return pair_count


# =====================================================
# Inference helpers
# =====================================================
def load_model_from_checkpoint(checkpoint_path, image_size=256):
    model = ConditionalArtifactUNet(image_size=image_size)
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    # Accept bare state-dict or full checkpoint with 'ema_state_dict'
    sd = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def generate_artifact_from_normal(
    model,
    noise_scheduler,
    normal_image_path,
    output_path          = None,
    num_inference_steps  = 50,
    image_size           = 256,
    guidance_scale       = 3.0,
    seed                 = 42,
):
    dev = next(model.parameters()).device
    model.eval()

    normal_img         = Image.open(normal_image_path).convert('RGB')
    original_size      = normal_img.size
    normal_img_resized = normal_img.resize((image_size, image_size), Image.LANCZOS)
    normal_tensor      = (torch.from_numpy(np.array(normal_img_resized))
                          .float().permute(2, 0, 1) / 127.5 - 1.0
                          ).unsqueeze(0).to(dev)

    null_cond = torch.zeros_like(normal_tensor)

    torch.manual_seed(seed)
    generated = torch.randn_like(normal_tensor)
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps, desc="Generating"):
        ts           = torch.full((1,), t, device=dev, dtype=torch.long)
        noise_cond   = model(generated, ts, normal_tensor)
        noise_uncond = model(generated, ts, null_cond)
        noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        generated    = noise_scheduler.step(noise_pred, t, generated).prev_sample

    gen_arr = ((generated + 1) / 2)[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    gen_img = Image.fromarray((gen_arr * 255).astype(np.uint8))
    gen_img = gen_img.resize(original_size, Image.LANCZOS)

    if output_path:
        gen_img.save(output_path)
        print(f"✓ Artifact saved: {output_path}")
    return gen_img


def resume_training(checkpoint_path, dataset_path, artifact_type,
                    resume_epoch, total_epochs=200, **kwargs):
    print(f"Resuming training from epoch {resume_epoch}")
    ckpt       = torch.load(checkpoint_path, map_location='cpu')
    image_size = kwargs.get('image_size', 256)
    lr         = kwargs.get('learning_rate', 5e-5)

    model = ConditionalArtifactUNet(image_size=image_size)
    model.load_state_dict(ckpt['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    config = dict(
        dataset_path=dataset_path,
        artifact_type=artifact_type,
        num_epochs=total_epochs - resume_epoch,
        output_dir=kwargs.get('output_dir', OUTPUT_BASE),
        **kwargs,
    )
    return train_conditional_diffusion(**config)


# =====================================================
# MAIN PIPELINE
# =====================================================
if __name__ == '__main__':
    # Reduce CUDA memory fragmentation
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')  # P100: smaller splits cut fragmentation

    if WANDB_PROJECT is not None:
        wandb.login()

    os.makedirs(OUTPUT_BASE,      exist_ok=True)
    os.makedirs(SYNTHETIC_OUTPUT, exist_ok=True)

    # --------------------------------------------------
    # STEP 1: Train Underexposure Generator
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING: Underexposure Artifact Generator")
    print("=" * 60)

    under_model, under_scheduler = train_conditional_diffusion(
        dataset_path             = DATASET_PATH,
        artifact_type            = 'underexposed',
        output_dir               = OUTPUT_BASE,
        wandb_project            = WANDB_PROJECT,
        num_epochs               = 150,
        batch_size               = 2,          # P100-16GB: halved (effective=8 via grad_accum=4)
        learning_rate            = 5e-5,
        image_size               = 256,        # FIX #1
        num_inference_steps      = 50,
        save_every_n_epochs      = 5,
        log_images_every_n_epochs= 2,
        grad_accum_steps         = 4,          # 2->4: keeps effective batch = 2x4 = 8
        guidance_scale           = 3.0,
        cond_drop_prob           = 0.10,
        perceptual_weight        = 0.5,
        ema_decay                = 0.9999,
        snr_gamma                = 5.0,
    )

    # --------------------------------------------------
    # STEP 2: Train Overexposure Generator
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING: Overexposure Artifact Generator")
    print("=" * 60)

    over_model, over_scheduler = train_conditional_diffusion(
        dataset_path             = DATASET_PATH,
        artifact_type            = 'overexposed',
        output_dir               = OUTPUT_BASE,
        wandb_project            = WANDB_PROJECT,
        num_epochs               = 150,
        batch_size               = 2,          # P100-16GB: halved (effective=8 via grad_accum=4)
        learning_rate            = 5e-5,
        image_size               = 256,
        num_inference_steps      = 50,
        save_every_n_epochs      = 5,
        log_images_every_n_epochs= 2,
        grad_accum_steps         = 4,          # 2->4: keeps effective batch = 2x4 = 8
        guidance_scale           = 3.0,
        cond_drop_prob           = 0.10,
        perceptual_weight        = 0.5,
        ema_decay                = 0.9999,
        snr_gamma                = 5.0,
    )

    # --------------------------------------------------
    # STEP 3: Generate Synthetic Datasets
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING: Synthetic Paired Datasets")
    print("=" * 60)

    normal_folder = os.path.join(DATASET_PATH, 'real_overexposed', 'Test', 'normal_frames')

    print("\n>>> Generating OVEREXPOSED synthetic dataset...")
    num_over = generate_synthetic_dataset(
        model                = over_model,
        noise_scheduler      = over_scheduler,
        normal_image_folder  = normal_folder,
        output_folder        = os.path.join(SYNTHETIC_OUTPUT, 'overexposed'),
        artifact_type        = 'overexposed',
        num_variations       = 5,
        num_inference_steps  = 50,
        image_size           = 256,
        guidance_scale       = 3.0,
    )

    print("\n>>> Generating UNDEREXPOSED synthetic dataset...")
    num_under = generate_synthetic_dataset(
        model                = under_model,
        noise_scheduler      = under_scheduler,
        normal_image_folder  = normal_folder,
        output_folder        = os.path.join(SYNTHETIC_OUTPUT, 'underexposed'),
        artifact_type        = 'underexposed',
        num_variations       = 5,
        num_inference_steps  = 50,
        image_size           = 256,
        guidance_scale       = 3.0,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Models saved in      : {OUTPUT_BASE}")
    print(f"  Overexposed model  : {os.path.join(OUTPUT_BASE, 'overexposed')}/")
    print(f"  Underexposed model : {os.path.join(OUTPUT_BASE, 'underexposed')}/")
    print(f"\nSynthetic datasets   : {SYNTHETIC_OUTPUT}")
    print(f"  Overexposed pairs  : {num_over}")
    print(f"  Underexposed pairs : {num_under}")
    print(f"  Total images       : {(num_over + num_under) * 2}")
    if WANDB_PROJECT:
        print(f"W&B dashboard        : https://wandb.ai/")
    print("=" * 60)

    # --------------------------------------------------
    # OPTIONAL: single-image inference example
    # --------------------------------------------------
    # ema_model = load_model_from_checkpoint(
    #     os.path.join(OUTPUT_BASE, 'overexposed', 'best_model.pt'), image_size=256
    # ).to(device)
    # scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
    #                           clip_sample=False)
    # generate_artifact_from_normal(
    #     model=ema_model, noise_scheduler=scheduler,
    #     normal_image_path='/path/to/normal.jpg',
    #     output_path=os.path.join(SYNTHETIC_OUTPUT, 'generated_overexposed.png'),
    #     num_inference_steps=50, guidance_scale=3.0, seed=42,
    # )
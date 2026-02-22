"""
Conditional Artifact Diffusion Model — DGX Server Version
==========================================================
Environment: Shared DGX Ubuntu, NVIDIA Driver 465.19.01, CUDA 11.3
Docker image: pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
Python: 3.8

PINNED DEPENDENCY VERSIONS (install exactly these):
----------------------------------------------------
# PyTorch stack — already in the Docker image, do NOT reinstall unless needed
  torch==1.11.0+cu113          (pre-installed)
  torchvision==0.12.0+cu113   (pre-installed)
  torchaudio==0.11.0           (pre-installed)

# Install the rest inside the container:
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
    packaging==23.1

RATIONALE (version choices):
-----------------------------
- torch 1.11.0+cu113    : locked by Docker image; last official cu113 build
- torchvision 0.12.0    : exact companion for torch 1.11.0
- diffusers 0.14.0      : last release before PyTorch >=1.13 became a hard dep;
                          DDPMScheduler & UNet2DModel APIs identical to later versions
- huggingface_hub 0.25.2: ROOT CAUSE FIX -- hub 0.26.0 permanently removed
                          `cached_download`, which diffusers 0.14.0 still imports
                          via dynamic_modules_utils.py; 0.25.2 is the last version
                          that kept it (deprecated but present); must be pinned
                          explicitly because pip will otherwise pull 0.26+ as a
                          transitive dep of transformers/diffusers
- accelerate 0.18.0     : requires torch >=1.10; fp16 AMP works with torch 1.11;
                          avoids 0.19+ which began requiring torch >=1.13 features
- transformers 4.27.4   : required by diffusers 0.14.0; compatible with torch 1.11
                          and huggingface_hub 0.25.2
- wandb 0.14.2          : last release before Python 3.8 support was dropped
                          (dropped in wandb 0.18.x); safe for Python 3.8
- Pillow 9.5.0          : last Pillow 9.x; Image.LANCZOS supported; stable on 3.8
- numpy 1.23.5          : last numpy 1.x fully compatible with Python 3.8 + torch 1.11
- tqdm 4.65.0           : stable, no breaking changes
- matplotlib 3.7.1      : Agg backend works; last version supporting Python 3.8 well
- packaging 23.1        : used by the runtime version guard

Run: python edm_t04_020226_1430_server.py
"""

# =====================================================
# CONFIGURATION — Edit these paths before running
# =====================================================

# Root folder of the Endo4IE dataset (synced via gdown)
DATASET_PATH = "/home/user/data/endo4ie"          # <-- EDIT THIS

# Where to save models, checkpoints and samples
OUTPUT_BASE  = "/home/user/outputs/endo_diffusion" # <-- EDIT THIS

# Synthetic dataset output folder
SYNTHETIC_OUTPUT = "/home/user/outputs/synthetic_datasets" # <-- EDIT THIS

# Weights & Biases project name (set to None to disable W&B)
WANDB_PROJECT = "endo-artifact-generation"        # <-- EDIT or set None

# =====================================================
# Imports
# =====================================================
import os
import sys
import numpy as np
from pathlib import Path
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

# Use non-interactive matplotlib backend (no display required)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Conditional W&B import
if WANDB_PROJECT is not None:
    import wandb

# =====================================================
# Runtime version guard — fail fast with a clear message
# =====================================================
def _check_versions():
    import accelerate as _acc
    import diffusers as _diff
    import huggingface_hub as _hfh

    # huggingface_hub >= 0.26 removed `cached_download` entirely; diffusers
    # 0.14.0 imports it at module load time, so this causes an immediate
    # ImportError before any training code runs. Check this first and hard-exit.
    hfh_ver = tuple(int(x) for x in _hfh.__version__.split(".")[:2])
    if hfh_ver >= (0, 26):
        print(
            f"\nFATAL: huggingface_hub {_hfh.__version__} is installed.\n"
            f"  huggingface_hub >= 0.26 removed `cached_download`, which\n"
            f"  diffusers==0.14.0 requires. Downgrade before running:\n\n"
            f"      pip install huggingface_hub==0.25.2\n"
        )
        sys.exit(1)

    checks = [
        ("Python",          sys.version_info[:2],
                            (3, 8),  (3, 8),  "3.8.x"),
        ("torch",           tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]),
                            (1, 11), (1, 11), "1.11.x"),
        ("diffusers",       tuple(int(x) for x in _diff.__version__.split(".")[:2]),
                            (0, 14), (0, 14), "0.14.x"),
        ("huggingface_hub", hfh_ver,
                            (0, 14), (0, 25), "0.14-0.25.x"),
        ("accelerate",      tuple(int(x) for x in _acc.__version__.split(".")[:2]),
                            (0, 12), (0, 20), "0.12-0.19.x"),
    ]
    warnings_list = []
    for name, got, lo, hi, expected in checks:
        if not (lo <= got <= hi):
            warnings_list.append(
                f"  ! {name}: got {'.'.join(str(x) for x in got)}, "
                f"expected {expected}"
            )
    if warnings_list:
        print("VERSION WARNING -- unexpected library versions detected:")
        for w in warnings_list:
            print(w)
        print("Training may still work, but results are not guaranteed.\n")
    else:
        print("All library versions match the pinned requirements.\n")

_check_versions()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPUs available: {torch.cuda.device_count()}")


# =====================================================
# Dataset
# =====================================================

class ConditionalArtifactDataset(Dataset):
    """
    Dataset for conditional artifact generation: p(x_artifact | x_normal).
    Loads paired data: normal images + corresponding real artifacts.

    Expected folder structure (example for overexposed):
      <base_path>/real_overexposed/<split>/normal_frames/
      <base_path>/real_overexposed/<split>/overexposed/
    where <split> is 'train', 'validation', or 'test' (case-insensitive match).
    """

    def __init__(self, base_path, split='train', artifact_type='overexposed', image_size=256):
        self.base_path = Path(base_path)
        self.split = split
        self.artifact_type = artifact_type
        self.image_size = image_size

        if artifact_type == 'overexposed':
            folder          = 'real_overexposed'
            artifact_folder = 'overexposed'
            normal_folder   = 'normal_frames'
        else:
            folder          = 'real_underexposed'
            artifact_folder = 'underexposed'
            normal_folder   = 'normal_frames'

        # Try exact split name first, then capitalised variants
        def _find_split_dir(parent: Path, name: str) -> Path:
            for candidate in [name, name.capitalize(), name.upper(), name.lower()]:
                p = parent / candidate
                if p.is_dir():
                    return p
            raise FileNotFoundError(
                f"Cannot find split directory '{name}' under {parent}. "
                f"Available: {[x.name for x in parent.iterdir() if x.is_dir()]}"
            )

        split_dir = _find_split_dir(self.base_path / folder, split)

        self.normal_path   = split_dir / normal_folder
        self.artifact_path = split_dir / artifact_folder

        for p in [self.normal_path, self.artifact_path]:
            if not p.is_dir():
                raise FileNotFoundError(f"Expected directory not found: {p}")

        _img_exts = ('.png', '.jpg', '.jpeg')
        self.normal_images   = sorted(f for f in os.listdir(self.normal_path)
                                      if f.lower().endswith(_img_exts))
        self.artifact_images = sorted(f for f in os.listdir(self.artifact_path)
                                      if f.lower().endswith(_img_exts))

        print(f"[{split}/{artifact_type}] "
              f"{len(self.normal_images)} normal + {len(self.artifact_images)} artifact images")

    def __len__(self):
        return min(len(self.normal_images), len(self.artifact_images))

    def _to_tensor(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0

    def __getitem__(self, idx):
        normal_name   = self.normal_images[idx]
        artifact_name = normal_name if normal_name in self.artifact_images \
                        else self.artifact_images[idx]

        normal_tensor   = self._to_tensor(self.normal_path   / normal_name)
        artifact_tensor = self._to_tensor(self.artifact_path / artifact_name)

        return {'normal': normal_tensor, 'artifact': artifact_tensor}


# =====================================================
# Model
# =====================================================

class ConditionalArtifactUNet(nn.Module):
    """
    Conditional Diffusion UNet: p(x_artifact | x_normal).
    Input channels = 6 (3 noisy artifact + 3 normal condition).
    """

    def __init__(self, image_size=256):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=6,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, noisy_artifact, timesteps, normal_condition):
        x = torch.cat([noisy_artifact, normal_condition], dim=1)
        return self.unet(x, timesteps).sample


# =====================================================
# Sample generation (saves PNG, no plt.show)
# =====================================================

@torch.no_grad()
def generate_and_save_samples(model, noise_scheduler, dataloader, output_dir,
                               epoch, num_inference_steps=50, num_samples=4):
    """Generate sample grid and save to disk (no display)."""
    model.eval()
    device = next(model.parameters()).device

    batch = next(iter(dataloader))
    normal_imgs   = batch['normal'][:num_samples].to(device)
    real_artifacts = batch['artifact'][:num_samples].to(device)

    generated = torch.randn_like(normal_imgs)
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        ts = torch.full((normal_imgs.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model(generated, ts, normal_imgs)
        generated  = noise_scheduler.step(noise_pred, t, generated).prev_sample

    # Denormalise
    normal_imgs    = (normal_imgs    + 1) / 2
    real_artifacts = (real_artifacts + 1) / 2
    generated      = (generated      + 1) / 2

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    row_titles = ['Normal (Condition)', 'Real Artifact', 'Generated Artifact']
    imgs_rows  = [normal_imgs, real_artifacts, generated]

    for row, (title, imgs) in enumerate(zip(row_titles, imgs_rows)):
        for col in range(num_samples):
            axes[row, col].imshow(imgs[col].cpu().permute(1, 2, 0).clamp(0, 1))
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'samples', f'epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Samples saved: {save_path}")


# =====================================================
# Training
# =====================================================

def train_conditional_diffusion(
    dataset_path,
    artifact_type='overexposed',
    output_dir='./conditional_diffusion',
    wandb_project=None,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    image_size=256,
    num_inference_steps=50,
    save_every_n_epochs=10,
    log_images_every_n_epochs=5,
    num_workers=4,
):
    """Train conditional diffusion model: p(artifact | normal)."""

    out_dir       = os.path.join(output_dir, artifact_type)
    samples_dir   = os.path.join(out_dir, 'samples')
    ckpt_dir      = os.path.join(out_dir, 'checkpoints')
    for d in [out_dir, samples_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    # W&B (optional)
    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"{artifact_type}-diffusion",
            config=dict(
                artifact_type=artifact_type,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                image_size=image_size,
                num_inference_steps=num_inference_steps,
                model='conditional-unet-diffusion',
            ),
        )

    accelerator = Accelerator(mixed_precision='fp16')

    train_dataset = ConditionalArtifactDataset(dataset_path, split='train',
                                               artifact_type=artifact_type, image_size=image_size)
    val_dataset   = ConditionalArtifactDataset(dataset_path, split='validation',
                                               artifact_type=artifact_type, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    model           = ConditionalArtifactUNet(image_size=image_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    global_step   = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            normal_imgs   = batch['normal']
            artifact_imgs = batch['artifact']

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (artifact_imgs.shape[0],), device=artifact_imgs.device
            ).long()

            noise          = torch.randn_like(artifact_imgs)
            noisy_artifacts = noise_scheduler.add_noise(artifact_imgs, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_artifacts, timesteps, normal_imgs)
                loss       = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if accelerator.is_main_process and use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr":   lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step":  global_step,
                })

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
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
            print(f"Epoch {epoch + 1}/{num_epochs} — "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if use_wandb:
                wandb.log({
                    "epoch":             epoch + 1,
                    "train/epoch_loss":  avg_train_loss,
                    "val/loss":          avg_val_loss,
                })

        # Sample visualisation (saved to disk only)
        if (epoch + 1) % log_images_every_n_epochs == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            generate_and_save_samples(
                unwrapped, noise_scheduler, val_loader, out_dir,
                epoch + 1, num_inference_steps
            )

        # Checkpoint
        if (epoch + 1) % save_every_n_epochs == 0 and accelerator.is_main_process:
            unwrapped      = accelerator.unwrap_model(model)
            ckpt_path      = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch':               epoch + 1,
                'model_state_dict':    unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':          avg_train_loss,
                'val_loss':            avg_val_loss,
                'config': {
                    'artifact_type':       artifact_type,
                    'image_size':          image_size,
                    'num_inference_steps': num_inference_steps,
                },
            }, ckpt_path)
            print(f"✓ Checkpoint saved: {ckpt_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss  = avg_val_loss
                best_path      = os.path.join(out_dir, 'best_model.pt')
                torch.save(unwrapped.state_dict(), best_path)
                print(f"✓ New best model saved: {best_path}")

    # Final model
    if accelerator.is_main_process:
        unwrapped   = accelerator.unwrap_model(model)
        final_path  = os.path.join(out_dir, 'final_model.pt')
        torch.save(unwrapped.state_dict(), final_path)
        print(f"\n✓ Final model saved: {final_path}")
        if use_wandb:
            wandb.finish()

    return model, noise_scheduler


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
    num_variations=5,
    num_inference_steps=50,
    image_size=256,
):
    """Generate complete synthetic paired dataset (Normal → Synthetic Artifacts)."""
    model.eval()
    device = next(model.parameters()).device

    os.makedirs(os.path.join(output_folder, 'normal'),                   exist_ok=True)
    os.makedirs(os.path.join(output_folder, f'synthetic_{artifact_type}'), exist_ok=True)

    _img_exts    = ('.png', '.jpg', '.jpeg')
    normal_images = sorted(f for f in os.listdir(normal_image_folder)
                           if f.lower().endswith(_img_exts))

    print(f"\nGenerating synthetic {artifact_type} dataset...")
    print(f"  Normal images      : {len(normal_images)}")
    print(f"  Variations / image : {num_variations}")
    print(f"  Total pairs        : {len(normal_images) * num_variations}")

    pair_count = 0

    for img_idx, img_name in enumerate(tqdm(normal_images, desc="Processing images")):
        normal_img        = Image.open(os.path.join(normal_image_folder, img_name)).convert('RGB')
        normal_img_resized = normal_img.resize((image_size, image_size), Image.LANCZOS)
        normal_tensor     = (torch.from_numpy(np.array(normal_img_resized))
                             .float().permute(2, 0, 1) / 127.5 - 1.0)
        normal_tensor     = normal_tensor.unsqueeze(0).to(device)

        for variation in range(num_variations):
            torch.manual_seed(img_idx * num_variations + variation)
            generated = torch.randn_like(normal_tensor)
            noise_scheduler.set_timesteps(num_inference_steps)

            for t in noise_scheduler.timesteps:
                ts         = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(generated, ts, normal_tensor)
                generated  = noise_scheduler.step(noise_pred, t, generated).prev_sample

            generated     = (generated + 1) / 2
            gen_img_arr   = generated[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            gen_img       = Image.fromarray((gen_img_arr * 255).astype(np.uint8))

            base_name  = os.path.splitext(img_name)[0]
            pair_name  = f"{base_name}_var{variation:02d}.png"

            normal_img.save(os.path.join(output_folder, 'normal', pair_name))
            gen_img.save(os.path.join(output_folder, f'synthetic_{artifact_type}', pair_name))
            pair_count += 1

    print(f"\n✓ Generated {pair_count} synthetic paired images")
    print(f"✓ Saved to: {output_folder}")
    return pair_count


# =====================================================
# Inference helpers
# =====================================================

def load_model_from_checkpoint(checkpoint_path, image_size=256):
    """Load a trained model from a .pt checkpoint file."""
    model      = ConditionalArtifactUNet(image_size=image_size)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Support both full checkpoint dicts and bare state dicts
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_artifact_from_normal(
    model,
    noise_scheduler,
    normal_image_path,
    output_path=None,
    num_inference_steps=50,
    image_size=256,
    seed=42,
):
    """Generate a single synthetic artifact from a normal image."""
    device = next(model.parameters()).device
    model.eval()

    normal_img         = Image.open(normal_image_path).convert('RGB')
    original_size      = normal_img.size
    normal_img_resized = normal_img.resize((image_size, image_size), Image.LANCZOS)
    normal_tensor      = (torch.from_numpy(np.array(normal_img_resized))
                          .float().permute(2, 0, 1) / 127.5 - 1.0)
    normal_tensor      = normal_tensor.unsqueeze(0).to(device)

    torch.manual_seed(seed)
    generated = torch.randn_like(normal_tensor)
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps, desc="Generating artifact"):
        ts         = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = model(generated, ts, normal_tensor)
        generated  = noise_scheduler.step(noise_pred, t, generated).prev_sample

    generated     = (generated + 1) / 2
    gen_img_arr   = generated[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    generated_img = Image.fromarray((gen_img_arr * 255).astype(np.uint8))
    generated_img = generated_img.resize(original_size, Image.LANCZOS)

    if output_path:
        generated_img.save(output_path)
        print(f"✓ Artifact saved to: {output_path}")

    return generated_img


def resume_training(
    checkpoint_path,
    dataset_path,
    artifact_type,
    resume_epoch,
    total_epochs=150,
    **kwargs,
):
    """Resume training from a saved checkpoint."""
    print(f"Resuming training from epoch {resume_epoch}")
    checkpoint  = torch.load(checkpoint_path, map_location='cpu')
    image_size  = kwargs.get('image_size', 128)
    lr          = kwargs.get('learning_rate', 1e-4)

    model = ConditionalArtifactUNet(image_size=image_size)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    remaining_epochs = total_epochs - resume_epoch
    config = {
        'dataset_path':  dataset_path,
        'artifact_type': artifact_type,
        'num_epochs':    remaining_epochs,
        'output_dir':    kwargs.get('output_dir', OUTPUT_BASE),
        **kwargs,
    }
    return train_conditional_diffusion(**config)


# =====================================================
# MAIN PIPELINE
# =====================================================

if __name__ == '__main__':

    # Optional: login to W&B (will prompt interactively or use WANDB_API_KEY env var)
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
        dataset_path=DATASET_PATH,
        artifact_type='underexposed',
        output_dir=OUTPUT_BASE,
        wandb_project=WANDB_PROJECT,
        num_epochs=100,
        batch_size=4,
        learning_rate=1e-4,
        image_size=256,
        num_inference_steps=50,
        save_every_n_epochs=5,
        log_images_every_n_epochs=1,
    )

    # --------------------------------------------------
    # STEP 2: Train Overexposure Generator
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING: Overexposure Artifact Generator")
    print("=" * 60)

    over_model, over_scheduler = train_conditional_diffusion(
        dataset_path=DATASET_PATH,
        artifact_type='overexposed',
        output_dir=OUTPUT_BASE,
        wandb_project=WANDB_PROJECT,
        num_epochs=100,
        batch_size=4,
        learning_rate=1e-4,
        image_size=256,
        num_inference_steps=50,
        save_every_n_epochs=10,
        log_images_every_n_epochs=5,
    )

    # --------------------------------------------------
    # STEP 3: Generate Synthetic Datasets
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING: Synthetic Paired Datasets")
    print("=" * 60)

    # Use test set normal images as generation source
    normal_folder = os.path.join(DATASET_PATH, 'real_overexposed', 'Test', 'normal_frames')

    print("\n>>> Generating OVEREXPOSED synthetic dataset...")
    num_over = generate_synthetic_dataset(
        model=over_model,
        noise_scheduler=over_scheduler,
        normal_image_folder=normal_folder,
        output_folder=os.path.join(SYNTHETIC_OUTPUT, 'overexposed'),
        artifact_type='overexposed',
        num_variations=5,
        num_inference_steps=50,
        image_size=256,
    )

    print("\n>>> Generating UNDEREXPOSED synthetic dataset...")
    num_under = generate_synthetic_dataset(
        model=under_model,
        noise_scheduler=under_scheduler,
        normal_image_folder=normal_folder,
        output_folder=os.path.join(SYNTHETIC_OUTPUT, 'underexposed'),
        artifact_type='underexposed',
        num_variations=5,
        num_inference_steps=50,
        image_size=256,
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
    print(f"\nSynthetic datasets generated:")
    print(f"  Overexposed pairs  : {num_over}")
    print(f"  Underexposed pairs : {num_under}")
    print(f"  Total images       : {(num_over + num_under) * 2}")
    print(f"\nCheckpoints and samples saved every epoch")
    if WANDB_PROJECT:
        print(f"Monitor training at  : https://wandb.ai/")
    print("=" * 60)

    # --------------------------------------------------
    # OPTIONAL: single-image inference example
    # Uncomment and adjust paths to use after training.
    # --------------------------------------------------
    # over_model_loaded = load_model_from_checkpoint(
    #     os.path.join(OUTPUT_BASE, 'overexposed', 'best_model.pt'),
    #     image_size=256
    # ).to(device)
    # scheduler = DDPMScheduler(num_train_timesteps=1000)
    # generate_artifact_from_normal(
    #     model=over_model_loaded,
    #     noise_scheduler=scheduler,
    #     normal_image_path='/path/to/normal_image.jpg',
    #     output_path=os.path.join(SYNTHETIC_OUTPUT, 'generated_overexposed.png'),
    #     num_inference_steps=50,
    #     seed=42,
    # )

    # --------------------------------------------------
    # OPTIONAL: resume training example
    # --------------------------------------------------
    # resume_training(
    #     checkpoint_path=os.path.join(OUTPUT_BASE, 'overexposed', 'checkpoints', 'checkpoint_epoch_50.pt'),
    #     dataset_path=DATASET_PATH,
    #     artifact_type='overexposed',
    #     resume_epoch=50,
    #     total_epochs=150,
    #     batch_size=4,
    #     learning_rate=5e-5,
    #     output_dir=OUTPUT_BASE,
    # )
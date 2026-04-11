import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from skimage.color import rgb2lab, lab2rgb
from packaging import version

# --- Configuration ---
class Config:
    def __init__(self, mode="local"):
        # Paths
        self.train_norm_path = "../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames" 
        self.train_target_path = "../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/overexposed_frames"
        self.output_dir = "../../../../projects/i2i_gem_v1/outputs"        
        # Hardware-Specific Tuning
        if mode == "dgx":
            self.resolution = 512
            self.batch_size = 2              # Absolute minimum to avoid OOM
            self.grad_accum_steps = 16       # Effective batch size = 16
            self.mixed_precision = "fp16"    # Standard for V100
            self.learning_rate = 1e-5        # Lower LR for Batch 1
            self.num_workers = 16
        else: # Local RTX 3050 (4GB)
            self.resolution = 256
            self.batch_size = 2
            self.mixed_precision = "fp16" 
            self.num_workers = 2
            self.learning_rate = 5e-5

        self.num_epochs = 100
        self.save_image_epochs = 5
        self.use_wandb = True
        self.enable_xformers = True

# --- Dataset Handling ---
class LabEndoscopyDataset(Dataset):
    """
    Separates L (Luminance) for training and keeps AB (Chrominance) for reconstruction.
    This ensures that medical color and high-frequency texture are preserved.
    """
    def __init__(self, norm_dir, target_dir, resolution=256):
        self.norm_images = [os.path.join(norm_dir, x) for x in os.listdir(norm_dir) if x.endswith(('.jpg', '.png'))]
        self.target_images = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if x.endswith(('.jpg', '.png'))]
        self.res = resolution

    def __len__(self):
        return max(len(self.norm_images), len(self.target_images))

    def _process_image(self, path):
        img = Image.open(path).convert("RGB").resize((self.res, self.res))
        lab = rgb2lab(np.array(img)).astype(np.float32)
        # Normalize L [0, 100] -> [-1, 1]
        l_chan = (torch.from_numpy(lab[:, :, 0:1]) / 50.0) - 1.0
        # AB channels are kept for reconstruction
        ab_chan = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1)
        return l_chan.permute(2, 0, 1), ab_chan

    def __getitem__(self, idx):
        norm_l, norm_ab = self._process_image(self.norm_images[idx % len(self.norm_images)])
        target_l, _ = self._process_image(self.target_images[idx % len(self.target_images)])
        
        return {
            "norm_l": norm_l,
            "target_l": target_l,
            "norm_ab": norm_ab
        }

# --- Trainer Class ---
class IlluminationDiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 1. Initialize Accelerator with Gradient Accumulation
        # This allows us to use a small batch size (e.g., 1) while simulating a larger one.
        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.grad_accum_steps 
        )
        
        # 2. Define the UNet2DModel
        # We use a slimmed channel count (starting at 32) to save VRAM at 512x512 resolution.
        # Attention is ONLY used at the 32x32 resolution level to prevent OOM.
        self.model = UNet2DModel(
            sample_size=cfg.resolution,
            in_channels=2,   # Noisy target L + Clean condition L
            out_channels=1,  # Predicts 1-channel noise
            block_out_channels=(32, 64, 128, 256, 512), 
            layers_per_block=2,
            down_block_types=(
                "DownBlock2D",      # 512x512
                "DownBlock2D",      # 256x256
                "DownBlock2D",      # 128x128
                "DownBlock2D",      # 64x64
                "AttnDownBlock2D",  # 32x32 (Small enough to fit global attention)
            ),
            up_block_types=(
                "AttnUpBlock2D",    # 32x32
                "UpBlock2D",        # 64x64
                "UpBlock2D",        # 128x128
                "UpBlock2D",        # 256x256
                "UpBlock2D",        # 512x512
            ),
        )

        # 3. Apply Memory Optimizations
        # Slicing computes attention in parts; xformers uses efficient kernels if available.
        if hasattr(self.model, "set_attention_slice"):
            self.model.set_attention_slice("auto")
        
        if cfg.enable_xformers:
            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception as e:
                # If xformers isn't installed in the container, we fall back gracefully.
                if self.accelerator.is_main_process:
                    print(f"[*] xformers not available, using standard attention: {e}")

        # 4. Initialize Scheduler and Optimizer
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg.learning_rate
        )
        
        # 5. Dataset and DataLoader
        # Note: We keep num_workers low to save CPU memory in Docker environments.
        dataset = LabEndoscopyDataset(
            cfg.train_norm_path, 
            cfg.train_target_path, 
            cfg.resolution
        )
        
        self.train_dataloader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True # Keeps batch dimensions consistent for accumulation
        )

        # 6. Prepare everything for the DGX environment
        # This handles device placement (.to(device)) and mixed precision scaling.
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        if self.accelerator.is_main_process:
            print(f"[*] Trainer initialized. Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self):
        # 1. Initialize Logging (Main Process Only)
        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project="Endo-Luminance-DGX",
                config={
                    "resolution": self.cfg.resolution,
                    "batch_size": self.cfg.batch_size,
                    "grad_accum_steps": self.cfg.grad_accum_steps,
                    "learning_rate": self.cfg.learning_rate,
                    "precision": self.cfg.mixed_precision
                }
            )

        best_loss = float('inf')
        global_step = 0

        print(f"[*] Starting Training. Total Epochs: {self.cfg.num_epochs}")
        
        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Progress bar for the main process
            progress_bar = tqdm(
                total=len(self.train_dataloader), 
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch}"
            )

            for step, batch in enumerate(self.train_dataloader):
                # 2. Gradient Accumulation Context Manager
                with self.accelerator.accumulate(self.model):
                    # Data Preparation
                    clean_l = batch["target_l"] # The L-channel we want to learn (Over/Under)
                    cond_l = batch["norm_l"]    # The Normal L-channel as condition
                    
                    # Diffusion Math
                    noise = torch.randn_like(clean_l)
                    bs = clean_l.shape[0]
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, 
                        (bs,), device=self.accelerator.device
                    ).long()

                    # Add noise to target L
                    noisy_l = self.noise_scheduler.add_noise(clean_l, noise, timesteps)
                    
                    # Concatenate noisy target + clean condition
                    # Shape: [B, 2, H, W]
                    model_input = torch.cat([noisy_l, cond_l], dim=1)
                    
                    # 3. Forward Pass
                    noise_pred = self.model(model_input, timesteps).sample
                    
                    # 4. Loss & Backward
                    loss = F.mse_loss(noise_pred, noise)
                    self.accelerator.backward(loss)
                    
                    # Optimizer Step (only happens every 'grad_accum_steps' steps)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update Progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                epoch_loss += loss.detach().item()

            # --- End of Epoch Logic ---
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            
            if self.accelerator.is_main_process:
                print(f"[*] Epoch {epoch} Average Loss: {avg_epoch_loss:.6f}")
                
                if self.cfg.use_wandb:
                    wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch}, step=global_step)

                # 5. Checkpointing (Save Best and Latest)
                # Unwrapping the model is necessary for saving when using Accelerator
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                # Latest
                latest_path = os.path.join(self.cfg.output_dir, "latest_model.pt")
                torch.save(unwrapped_model.state_dict(), latest_path)

                # Best
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_path = os.path.join(self.cfg.output_dir, "best_model.pt")
                    torch.save(unwrapped_model.state_dict(), best_path)
                    print(f"    --> New Best Loss! Model saved to {best_path}")

                # 6. Periodic Visual Sampling
                if epoch % self.cfg.save_image_epochs == 0:
                    print(f"[*] Generating Visual Sample for Epoch {epoch}...")
                    # We pass the same batch to see progress on the same images
                    self.save_samples(batch, epoch, global_step)

        if self.accelerator.is_main_process and self.cfg.use_wandb:
            wandb.finish()

    @torch.no_grad()
    def save_samples(self, batch, epoch):
        self.model.eval()
        # Take the first image in the batch
        norm_l = batch["norm_l"][0:1]
        norm_ab = batch["norm_ab"][0:1]
        
        # Start from pure noise
        sample = torch.randn_like(norm_l).to(self.accelerator.device)
        
        for t in self.noise_scheduler.timesteps:
            model_input = torch.cat([sample, norm_l], dim=1)
            model_output = self.model(model_input, t).sample
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
            
        # Reconstruct RGB
        sample_l = ((sample.clamp(-1, 1) + 1.0) * 50.0).cpu().numpy().squeeze()
        ab = norm_ab.cpu().numpy().squeeze()
        
        lab_img = np.zeros((self.cfg.resolution, self.cfg.resolution, 3))
        lab_img[:,:,0] = sample_l
        lab_img[:,:,1:] = ab.transpose(1, 2, 0)
        
        rgb_img = (lab2rgb(lab_img) * 255).astype(np.uint8)
        
        out_path = os.path.join(self.cfg.output_dir, f"sample_e{epoch}.png")
        Image.fromarray(rgb_img).save(out_path)
        if self.cfg.use_wandb:
            wandb.log({"visual_sample": wandb.Image(rgb_img)}, step=epoch)

# --- Main Execution ---
if __name__ == "__main__":
    # Choose 'local' for RTX 3050 or 'dgx' for DGX
    target_env = "dgx" # Change to "dgx" when moving to the cluster
    
    cfg = Config(mode=target_env)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    trainer = IlluminationDiffusionTrainer(cfg)
    trainer.train()
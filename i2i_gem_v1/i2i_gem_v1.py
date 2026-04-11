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
            self.batch_size = 16
            self.mixed_precision = "fp16"
            self.num_workers = 8
            self.learning_rate = 2e-4
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
        self.accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
        
        # Architecture: 2 channels in (noisy_target_L + condition_norm_L), 1 out (noise_L)
        self.model = UNet2DModel(
            sample_size=cfg.resolution,
            in_channels=2, 
            out_channels=1,
            block_out_channels=(64, 128, 256, 512),
            layers_per_block=2,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

        if cfg.enable_xformers:
            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Xformers not available: {e}")

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        
        # Dataset & Loader
        dataset = LabEndoscopyDataset(cfg.train_norm_path, cfg.train_target_path, cfg.resolution)
        self.train_dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, 
            num_workers=cfg.num_workers, pin_memory=True
        )

        # Prepare for Accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

    def train(self):
        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="Endo-I2I-Diffusion-Luminance")

        best_loss = float('inf')
        
        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            epoch_loss = 0
            for batch in self.train_dataloader:
                clean_l = batch["target_l"]
                cond_l = batch["norm_l"]
                
                # Sample noise and timesteps
                noise = torch.randn_like(clean_l)
                timesteps = torch.randint(0, 1000, (clean_l.shape[0],), device=self.accelerator.device).long()
                
                # Add noise to target luminance (L)
                noisy_l = self.noise_scheduler.add_noise(clean_l, noise, timesteps)
                
                # Concatenate condition (Normal L) to noisy target
                model_input = torch.cat([noisy_l, cond_l], dim=1)
                
                # Predict noise
                prediction = self.model(model_input, timesteps).sample
                
                # Loss calculation
                loss = F.mse_loss(prediction, noise)
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                progress_bar.update(1)
                epoch_loss += loss.detach().item()

            avg_loss = epoch_loss / len(self.train_dataloader)
            if self.accelerator.is_main_process:
                print(f"[*] Epoch {epoch} Avg Loss: {avg_loss:.6f}")
                if self.cfg.use_wandb:
                    wandb.log({"loss": avg_loss, "epoch": epoch})
                
                # Save Best Model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.accelerator.save(self.model.state_dict(), os.path.join(self.cfg.output_dir, "best_model.pt"))
                
                # Periodically generate samples
                if epoch % self.cfg.save_image_epochs == 0:
                    self.save_samples(batch, epoch)

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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDIMScheduler
from accelerate import Accelerator
from skimage.color import rgb2lab, lab2rgb

# --- 1. Configuration System ---
class Config:
    def __init__(self, mode="dgx"):
        # Folder Paths - ADJUST THESE TO YOUR DIRECTORY STRUCTURE
        self.train_norm_path = "../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames" 
        self.train_target_path = "../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/overexposed_frames"
        self.output_dir = "../../../../projects/i2i_gem_v1/outputs"  
        
        # Hardware-Specific Configuration
        if mode == "dgx":
            self.resolution = 512
            self.batch_size = 1              # Keep at 1 for 512x512 on 16GB
            self.grad_accum_steps = 16       # Effective Batch Size = 16
            self.mixed_precision = "fp16"    # Standard for V100/A100/H100
            self.num_workers = 4
            self.learning_rate = 5e-5
        else: # Local RTX 3050 (4GB)
            self.resolution = 256
            self.batch_size = 1
            self.grad_accum_steps = 4
            self.mixed_precision = "fp16" 
            self.num_workers = 0
            self.learning_rate = 2e-5

        self.num_epochs = 100
        self.save_image_epochs = 5
        self.use_wandb = True
        self.enable_xformers = True

# --- 2. Dataset Logic (Luminance Isolation) ---
class LabEndoscopyDataset(Dataset):
    def __init__(self, norm_dir, target_dir, resolution=256):
        self.norm_images = [os.path.join(norm_dir, x) for x in os.listdir(norm_dir) if x.endswith(('.jpg', '.png'))]
        self.target_images = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if x.endswith(('.jpg', '.png'))]
        self.res = resolution

    def __len__(self):
        return max(len(self.norm_images), len(self.target_images))

    def _process_image(self, path):
        img = Image.open(path).convert("RGB").resize((self.res, self.res))
        lab = rgb2lab(np.array(img)).astype(np.float32)
        # Normalize L [0, 100] -> [-1, 1] for Diffusion
        l_chan = (torch.from_numpy(lab[:, :, 0:1]) / 50.0) - 1.0
        ab_chan = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1)
        return l_chan.permute(2, 0, 1), ab_chan

    def __getitem__(self, idx):
        norm_l, norm_ab = self._process_image(self.norm_images[idx % len(self.norm_images)])
        target_l, _ = self._process_image(self.target_images[idx % len(self.target_images)])
        return {"norm_l": norm_l, "target_l": target_l, "norm_ab": norm_ab}

# --- 3. The Trainer Class ---
class IlluminationDiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.grad_accum_steps
        )
        
        # SLIMMED ARCHITECTURE: Fits in 16GB VRAM at 512x512
        self.model = UNet2DModel(
            sample_size=cfg.resolution,
            in_channels=2, # Noisy Target L + Clean Normal L
            out_channels=1,
            block_out_channels=(32, 64, 128, 256, 512), 
            layers_per_block=2,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

        # Memory Optimization
        if hasattr(self.model, "set_attention_slice"):
            self.model.set_attention_slice("auto")
        
        if cfg.enable_xformers:
            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        # Faster Scheduler
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        
        dataset = LabEndoscopyDataset(cfg.train_norm_path, cfg.train_target_path, cfg.resolution)
        self.train_dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, 
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

    def train(self):
        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="Endo-I2I-Luminance")

        best_loss = float('inf')
        global_step = 0

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    clean_l = batch["target_l"]
                    cond_l = batch["norm_l"]
                    noise = torch.randn_like(clean_l)
                    timesteps = torch.randint(0, 1000, (clean_l.shape[0],), device=self.accelerator.device).long()
                    
                    noisy_l = self.noise_scheduler.add_noise(clean_l, noise, timesteps)
                    model_input = torch.cat([noisy_l, cond_l], dim=1)
                    
                    prediction = self.model(model_input, timesteps).sample
                    loss = F.mse_loss(prediction, noise)
                    
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                epoch_loss += loss.detach().item()

            avg_loss = epoch_loss / len(self.train_dataloader)
            if self.accelerator.is_main_process:
                print(f"[*] Epoch {epoch} Avg Loss: {avg_loss:.6f}")
                if self.cfg.use_wandb:
                    wandb.log({"loss": avg_loss, "epoch": epoch}, step=global_step)

                # Save Checkpoints
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), os.path.join(self.cfg.output_dir, "latest_model.pt"))
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(unwrapped_model.state_dict(), os.path.join(self.cfg.output_dir, "best_model.pt"))

                # Sampling
                if epoch % self.cfg.save_image_epochs == 0:
                    self.save_samples(batch, epoch, global_step)

    @torch.no_grad()
    def save_samples(self, batch, epoch, global_step):
        self.model.eval()
        norm_l = batch["norm_l"][0:1].to(self.accelerator.device)
        norm_ab = batch["norm_ab"][0:1]
        
        sample = torch.randn_like(norm_l).to(self.accelerator.device)
        
        # SPEED HACK: Use 50 steps instead of 1000 for preview
        self.noise_scheduler.set_timesteps(50)
        for t in self.noise_scheduler.timesteps:
            model_input = torch.cat([sample, norm_l], dim=1)
            noise_pred = self.model(model_input, t).sample
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
        
        self.noise_scheduler.set_timesteps(1000) # Reset
            
        sample_l = ((sample.clamp(-1, 1) + 1.0) * 50.0).cpu().numpy().squeeze()
        ab = norm_ab.cpu().numpy().squeeze()
        
        lab_img = np.zeros((self.cfg.resolution, self.cfg.resolution, 3))
        lab_img[:,:,0] = sample_l
        lab_img[:,:,1:] = ab.transpose(1, 2, 0)
        
        rgb_img = (lab2rgb(lab_img) * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(os.path.join(self.cfg.output_dir, f"sample_e{epoch}.png"))
        if self.cfg.use_wandb:
            wandb.log({"visual_sample": wandb.Image(rgb_img)}, step=global_step)

# --- 4. Main Execution ---
if __name__ == "__main__":
    cfg = Config(mode="dgx") # Ensure this is "dgx" for cluster training
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer = IlluminationDiffusionTrainer(cfg)
    trainer.train()
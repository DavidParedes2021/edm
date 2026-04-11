import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from skimage.color import rgb2lab, lab2rgb
import wandb

# --- Configuration ---
class Config:
    def __init__(self):
        self.train_norm_path = "../curated_edm2020_classified/normal_frames"
        self.train_over_path = "../curated_edm2020_classified/overexposed_frames"
        self.train_under_path = "../curated_edm2020_classified/underexposed_frames"
        self.output_dir = "./outputs"
        self.resolution = 256
        self.batch_size = 4 if torch.cuda.get_device_properties(0).total_memory < 8e9 else 16
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.save_image_epochs = 5
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "fp16" # Crucial for RTX 3050
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_wandb = True

# --- Dataset Handling ---
class EndoscopyDataset(Dataset):
    def __init__(self, norm_dir, target_dir, resolution=256):
        self.norm_images = [os.path.join(norm_dir, x) for x in os.listdir(norm_dir)]
        self.target_images = [os.path.join(target_dir, x) for x in os.listdir(target_dir)]
        self.res = resolution

    def __len__(self):
        return max(len(self.norm_images), len(self.target_images))

    def __getitem__(self, idx):
        # Unpaired loading logic
        n_path = self.norm_images[idx % len(self.norm_images)]
        t_path = self.target_images[idx % len(self.target_images)]
        
        norm_img = Image.open(n_path).convert("RGB").resize((self.res, self.res))
        target_img = Image.open(t_path).convert("RGB").resize((self.res, self.res))

        # Convert to LAB
        norm_lab = rgb2lab(np.array(norm_img)).astype(np.float32)
        target_lab = rgb2lab(np.array(target_img)).astype(np.float32)

        # Normalize L to [-1, 1] for Diffusion
        # L is 0-100 in skimage
        norm_l = (torch.from_numpy(norm_lab[:, :, 0:1]) / 50.0) - 1.0
        target_l = (torch.from_numpy(target_lab[:, :, 0:1]) / 50.0) - 1.0
        
        # Permute to [C, H, W]
        return {
            "norm_l": norm_l.permute(2, 0, 1),
            "target_l": target_l.permute(2, 0, 1),
            "norm_ab": torch.from_numpy(norm_lab[:, :, 1:]).permute(2, 0, 1)
        }

# --- Loss Functions ---
def exposure_loss(generated_l, target_l):
    """Encourages the generated L-channel to match the mean/std of the target domain."""
    return F.mse_loss(generated_l.mean(), target_l.mean()) + \
           F.mse_loss(generated_l.std(), target_l.std())

def edge_preservation_loss(generated_l, original_l):
    """Sobel filter to ensure mucosal edges remain sharp."""
    def sobel(x):
        gx = F.conv2d(x, torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device).float().view(1,1,3,3))
        gy = F.conv2d(x, torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device).float().view(1,1,3,3))
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    return F.l1_loss(sobel(generated_l), sobel(original_l))

# --- Pipeline ---
class IlluminationDiffusion:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Switched to UNet2DModel for pure spatial concatenation
        self.model = UNet2DModel(
            sample_size=cfg.resolution,
            in_channels=2,  # 1 channel for noisy target L + 1 channel for normal conditioning L
            out_channels=1, # Output is just the predicted noise for the L channel
            block_out_channels=(64, 128, 256, 256),
            layers_per_block=2,
            # Explicitly define spatial blocks (no cross-attention)
            down_block_types=(
                "DownBlock2D", 
                "AttnDownBlock2D", 
                "AttnDownBlock2D", 
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D"
            ),
        ).to(cfg.device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        
        if cfg.use_wandb:
            wandb.init(project="Endo-Illumination-Diffusion")

    def train_step(self, batch, mode="over"):
        self.model.train()
        clean_l = batch["target_l"].to(self.cfg.device)
        cond_l = batch["norm_l"].to(self.cfg.device)
        
        noise = torch.randn_like(clean_l)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (clean_l.shape[0],), device=self.cfg.device).long()
        
        # Add noise to target luminance
        noisy_l = self.noise_scheduler.add_noise(clean_l, noise, timesteps)
        
        # Concatenate noisy target and clean condition along the channel dimension (Dim 1)
        # Shape becomes [batch_size, 2, height, width]
        model_input = torch.cat([noisy_l, cond_l], dim=1) 
        
        # Forward pass (UNet2DModel just takes sample and timestep)
        prediction = self.model(model_input, timesteps).sample 
        
        # Diffusion Loss
        loss_diff = F.mse_loss(prediction, noise)
        
        total_loss = loss_diff
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss.item()

    @torch.no_grad()
    def generate_sample(self, norm_l, norm_ab):
        self.model.eval()
        l_shape = norm_l.shape
        
        # Start from pure noise
        img = torch.randn(l_shape).to(self.cfg.device)
        norm_l = norm_l.to(self.cfg.device)
        
        for t in tqdm(self.noise_scheduler.timesteps):
            # MUST concatenate the condition during inference as well!
            model_input = torch.cat([img, norm_l], dim=1)
            
            model_output = self.model(model_input, t).sample
            img = self.noise_scheduler.step(model_output, t, img).prev_sample
            
        # Post-process back to RGB
        img = (img + 1.0) * 50.0 # Back to [0, 100]
        img = img.clamp(0, 100).cpu().numpy().squeeze()
        ab = norm_ab.cpu().numpy()
        
        # Reconstruct LAB image
        combined_lab = np.zeros((self.cfg.resolution, self.cfg.resolution, 3))
        combined_lab[:,:,0] = img
        combined_lab[:,:,1:] = ab.transpose(1, 2, 0)
        
        rgb_out = lab2rgb(combined_lab)
        return (rgb_out * 255).astype(np.uint8)


# --- Main Runner ---
if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Initialize Dataset for Overexposure
    dataset_over = EndoscopyDataset(cfg.train_norm_path, cfg.train_over_path, cfg.resolution)
    loader_over = DataLoader(dataset_over, batch_size=cfg.batch_size, shuffle=True)
    
    trainer = IlluminationDiffusion(cfg)
    
    print(f"Starting training on {cfg.device}...")
    best_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0
        for batch in loader_over:
            loss = trainer.train_step(batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(loader_over)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if cfg.use_wandb:
            wandb.log({"loss": avg_loss, "epoch": epoch})

        # Save Checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(trainer.model.state_dict(), f"{cfg.output_dir}/best_model.pt")
        
        torch.save(trainer.model.state_dict(), f"{cfg.output_dir}/latest_model.pt")

        # Periodically save samples
        if epoch % cfg.save_image_epochs == 0:
            sample_batch = next(iter(loader_over))
            res = trainer.generate_sample(sample_batch["norm_l"][0:1], sample_batch["norm_ab"][0:1])
            Image.fromarray(res).save(f"{cfg.output_dir}/sample_epoch_{epoch}.png")
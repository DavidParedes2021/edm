# Illumination Artifact Diffusion Pipeline

Unpaired → Paired synthetic dataset generator using a CycleGAN-inspired
DDPM backbone with ControlNet structural conditioning.

## Project layout

```
i2i_diffusion/
├── README.md
├── configs/
│   └── train_config.yaml        # all hyperparameters in one place
├── data/
│   ├── dataset.py               # UnpairedIlluminationDataset
│   └── transforms.py            # augmentations + histogram matching
├── models/
│   ├── unet_conditioned.py      # class-conditioned U-Net (diffusers 0.14 compatible)
│   ├── controlnet_lite.py       # lightweight structural hint injector
│   ├── discriminator.py         # PatchGAN discriminator
│   └── ema.py                   # EMA weight averaging
├── losses/
│   ├── perceptual.py            # LPIPS via VGG16 (torchvision 0.12)
│   ├── ssim_loss.py             # structural SSIM on gradient maps
│   └── cycle_loss.py            # cycle-consistency + identity helpers
├── training/
│   ├── trainer.py               # main training loop
│   └── noise_scheduler.py       # DDPM scheduler wrapper (diffusers 0.14 API)
├── inference/
│   └── generate_paired.py       # generates paired (Normal, Over, Under) triplets
├── utils/
│   ├── logging_utils.py         # wandb + local checkpointing
│   └── metrics.py               # FID, LPIPS-to-source, SSIM-gradient, SNR
└── train.py                     # entry-point
```

## Quick start (single GPU docker container)

```bash
# 1. Install deps (already done in your container)
# 2. Prepare data directories:
#    data/raw/normal/   *.png / *.jpg
#    data/raw/over/     *.png / *.jpg
#    data/raw/under/    *.png / *.jpg

# 3. Train
python train.py --config configs/train_config.yaml

# 4. Generate paired dataset
python inference/generate_paired.py \
    --checkpoint runs/<run_id>/checkpoints/best.pt \
    --source_dir data/raw/normal \
    --output_dir data/paired_synthetic \
    --guidance_scale 5.0
```

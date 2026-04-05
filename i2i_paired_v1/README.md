# Diffusion-Based Illumination Artifact Generation

## Architecture Decision

**Chosen: Palette-style I2I + ControlNet-inspired conditioning**

### Why NOT vanilla ControlNet
ControlNet was designed for text-to-image (SD 1.x backbone). Adapting it for pure I2I without text prompts adds unnecessary complexity and VRAM overhead. Our task is a *deterministic* style transfer (Normal → Over/Under-exposed), not open-ended generation.

### Why Palette-style I2I diffusion
- Palette (Ho et al., 2022) conditions the denoising U-Net on a *concatenated* source image (normal frame), giving pixel-aligned spatial control — exactly what we need.
- The model learns a **direct pixel-to-pixel mapping** in the diffusion reverse process.
- No text encoder needed → saves ~1 GB VRAM.
- Sharpness fix: We replace the simple MSE noise loss with a **perceptual + MSE hybrid loss** and use **v-prediction** parameterization.
- Exposure fix: We add a lightweight **exposure conditioning MLP** that embeds a scalar target exposure level, injected into every U-Net residual block via AdaGN (Adaptive Group Normalization).

## Conditioning Strategy

```
Normal Frame  ──► [Concat with noisy target, channel-wise] ──► U-Net
Exposure Label ──► [MLP → scalar embedding] ──► AdaGN in every ResBlock
```

The exposure label is a float in `[-1.0, +1.0]`:
- `-1.0` = maximum underexposure
- `+1.0` = maximum overexposure
- `0.0`  = normal (identity)

## Sharpness & Exposure Fixes
1. **v-prediction** instead of ε-prediction: predicts velocity = less blurry at low NFE.
2. **Perceptual loss** (VGG16 features) on x0-predictions during training.
3. **Exposure-aware data augmentation**: random exposure strength in [0.5, 1.0] × label.
4. **DDPM → DDIM** at inference: deterministic, fewer steps, sharper outputs.
5. **EMA** model weights for smoother training convergence.

## Evaluation Metrics
- **SSIM** – structural similarity to real over/underexposed targets
- **PSNR** – pixel fidelity
- **LPIPS** – perceptual sharpness (lower = better)
- **Exposure Bias Score** – mean luminance deviation from GT
- **FID** – distribution realism of generated artifacts

## Dataset Layout Expected
```
real_underexposed/train/underexposed/  ← artifact frames
real_underexposed/train/normal_frames/ ← paired normal frames
real_overexposed/train/overexposed/
real_overexposed/train/normal_frames/
```

## Quick Start

```bash
# Verify training starts (laptop, 4GB GPU)
python scripts/train.py --config configs/debug.yaml

# Full training (DGX, 16GB GPU)
python scripts/train.py --config configs/train.yaml

# Generate synthetic dataset
python scripts/generate.py --config configs/train.yaml --checkpoint outputs/best.pt
```

## Installation
```bash
pip install diffusers==0.14.0 huggingface_hub==0.25.2 accelerate==0.18.0 \
            transformers==4.27.4 wandb==0.14.2 Pillow==9.5.0 \
            numpy==1.23.5 tqdm==4.65.0 matplotlib==3.7.1 packaging==23.1 \
            torchvision==0.12.0+cu113
```

# 1. Create dummy data (laptop, no dataset needed)
python scripts/create_dummy_data.py

# 2. Smoke test (laptop 4GB GPU)
python scripts/train.py --config configs/debug.yaml

# 3. Deploy to DGX
git add . && git commit -m "initial" && git push
# on DGX:
git pull && python scripts/train.py --config configs/train.yaml

# 4. Generate synthetic dataset
python scripts/generate.py \
    --config configs/train.yaml \
    --checkpoint outputs/train/checkpoints/best.pt \
    --input_dir path/to/normal_frames/ \
    --output_dir synthetic_dataset/ \
    --exposure both --strength 0.9

# 5. Evaluate
python evaluation/metrics.py \
    --generated synthetic_dataset/overexposed/ \
    --reference real_overexposed/test/overexposed/ \
    --normal    real_overexposed/test/normal_frames/

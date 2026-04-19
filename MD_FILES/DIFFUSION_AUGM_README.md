# Paired Luminance Diffusion — Exposure Augmentation

## Architecture

**Palette-style conditional DDPM on the L channel only.**

The UNet takes 2 input channels — `concat(noisy_target_L, source_L)` — and predicts the noise added to `target_L`. A class embedding distinguishes overexposed (0) from underexposed (1). The source L channel is a pixel-level condition: the model sees exactly which structures exist and only learns the luminance shift pattern.

At inference, texture is preserved via frequency decomposition: the diffusion produces the low-frequency exposure envelope, while the original image's high-pass detail is re-injected unchanged. A and B chrominance channels are never processed by the model.

## Why This Works

The original unpaired approach failed because:
1. **SDEdit destroys texture** — denoising reconstructs pixels from noise, losing high-frequency detail (measured: 23% texture energy loss, 0.55 correlation).
2. **Weak class separation** — with 1100 unpaired, noisy-labelled images, CFG couldn't separate the two domains (measured: 0.2 L shift, essentially zero).

The paired approach fixes both:
1. **Paired supervision** — the model trains against exact target L channels, so it learns precisely the shift pattern that `exposure_augment.py` produces.
2. **Texture injection at inference** — the diffusion only provides the low-frequency envelope; original texture is physically preserved.
3. **Stochastic variation** — unlike the deterministic script, each inference run produces slightly different results due to the noise-to-signal path, giving a richer synthetic dataset.

## Should the model use the overexposed/underexposed folders?

**No.** Those folders are not needed. The training targets come from `exposure_augment.py` applied to normal frames, producing perfectly paired (normal → target) data. The original over/underexposed folders are noisy, unpaired, and inconsistently labelled — using them would reintroduce the problems that caused the original failures.

## Pipeline

```
Step 1: Generate pairs        Step 2: Train           Step 3: Generate
┌─────────────────┐          ┌──────────────┐        ┌─────────────────┐
│ normal frames   │──────────│ conditional  │────────│ new normal frame │
│ + augment.py    │  paired  │ DDPM on L    │ model  │ + DDIM denoise  │
│ → target L      │  data    │ channel      │        │ + texture inject │
└─────────────────┘          └──────────────┘        └─────────────────┘
```

## Usage

```bash
# 1. Generate paired training data
python generate_pairs.py \
    --normal_dir ./data/normal \
    --output_dir ./data/pairs \
    --strength 0.85 --shift_magnitude 50.0

# For more training diversity (3 variants per image = 1512 pairs per domain):
python generate_pairs.py \
    --normal_dir ./data/normal \
    --output_dir ./data/pairs \
    --num_variants 3

# 2. Train (DGX 16 GB)
python diffusion_train.py --config diffusion_config.yaml

# 3. Generate
python diffusion_inference.py \
    --config diffusion_config.yaml \
    --checkpoint ./output_diffusion/checkpoints/best.pt

# 4. Evaluate
python evaluate.py \
    --normal_dir ./data/normal \
    --generated_dir ./output_diffusion/generated/overexposed
```

## File Overview

| File | Purpose |
|---|---|
| `exposure_augment.py` | Deterministic cluster-based augmentation (approved baseline) |
| `generate_pairs.py` | Creates paired (normal_L → target_L) training data |
| `diffusion_config.yaml` | All hyperparameters |
| `diffusion_dataset.py` | Paired L-channel dataset + inference dataset |
| `diffusion_train.py` | Conditional DDPM training loop |
| `diffusion_inference.py` | DDIM generation with texture injection |
| `losses.py` | Sobel edge loss for sharpness |
| `evaluate.py` | SSIM, PSNR, brightness shift, histogram metrics |

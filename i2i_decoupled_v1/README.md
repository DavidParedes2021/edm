# Diffusion-Based Illumination Artifact Generation Pipeline

Generates **synthetic paired datasets** of overexposed and underexposed frames
from normal frames, using a class-conditional latent diffusion model that
operates exclusively in the luminance (Y) channel of the YCbCr color space.

---

## Architecture: YCbCr-Conditioned Luminance Diffusion (YCLDI)

### Why YCbCr — Hypothesis Validation

Processing exclusively in the **Y channel (luminance)** is both valid and optimal:

| Channel | Encodes | Affected by exposure? |
|---|---|---|
| Y (luminance) | Brightness | Yes — direct target |
| Cb (blue chroma) | Hue/saturation | No — preserved |
| Cr (red chroma) | Hue/saturation | No — preserved |

Diffusing only Y and recombining with the original Cb/Cr guarantees:
- **Color fidelity** — no hue/saturation drift in outputs
- **Structural preservation** — chrominance edges intact
- **3x efficiency** — 1-channel UNet instead of 3-channel

### Model: Conditional UNet with AdaGN

```
Input: Normal frame -> YCbCr -> Y channel
         |
cat(Y_noisy, Y_normal) -> Conv -> [Encoder: ResBlocks + Self-Attention]
                                       | AdaGN at every block
                               [Bottleneck: Self-Attention]
                                       |
                            [Decoder: ResBlocks + skip connections]
                                       |
                               Conv -> epsilon_predicted (B, 1, H, W)
```

**AdaGN (Adaptive Group Normalization)** is the control mechanism:
each residual block receives a conditioning vector `(timestep + exposure_class)`
and applies learned `scale` and `shift` to its group norm output.
This injects the exposure target at every spatial scale simultaneously.

### Control: Classifier-Free Guidance (CFG)

- Two classes: `OVER=0`, `UNDER=1`, `NULL=2` (for unconditional path)
- 15% of training batches use the `NULL` class (CFG dropout)
- At inference: `epsilon = epsilon_uncond + gamma * (epsilon_cond - epsilon_uncond)`
- `gamma=5-7` recommended; higher values produce stronger exposure effects

### Addressing Previous Failure Modes

| Problem | Solution |
|---|---|
| Blurry outputs | GradientDifferenceLoss on finite differences + VGG perceptual loss |
| Imperceptible exposure | HistogramMatchingLoss (Wasserstein-1 on CDF) forces output distribution to match reference exposure statistics |

---

## Project Structure

```
illumination_diffusion/
├── train.py                     # Main training loop
├── inference.py                 # Generate synthetic paired dataset
├── configs/
│   ├── laptop_debug.yaml        # 4GB GPU - smoke test
│   └── dgx_train.yaml           # 16GB DGX - full training
├── models/
│   ├── unet.py                  # Conditional UNet (AdaGN)
│   ├── embeddings.py            # Sinusoidal timestep + exposure class embedding
│   └── ema.py                   # EMA weight averaging
├── data/
│   └── dataset.py               # DataLoader - accepts direct paths
├── utils/
│   ├── color_space.py           # RGB <-> YCbCr (tensor-native, no numpy in forward)
│   ├── diffusion.py             # DDPM scheduler + DDIM sampler + CFG
│   ├── losses.py                # L1 + VGG perceptual + gradient + histogram
│   └── device.py                # Centralized device management
├── evaluation/
│   └── metrics.py               # PSNR, SSIM, EVS, Histogram KL
├── scripts/
│   └── prepare_dataset.py       # Validate dataset using config paths
├── setup.sh                     # Environment install (auto-detects CPU/GPU)
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install

```bash
# Auto-detects GPU and installs the right torch build
bash setup.sh

# Force CPU only (laptop without CUDA):
bash setup.sh --cpu

# Force DGX cu113 build:
bash setup.sh --dgx
```

### 2. Configure paths

Edit `configs/laptop_debug.yaml` or `configs/dgx_train.yaml`:

```yaml
data:
  normal_path: "/path/to/your/normal/frames"     # 504 frames
  over_path:   "/path/to/your/over/frames"       # 281 frames
  under_path:  "/path/to/your/under/frames"      # 817 frames

output:
  base_dir:       "/path/to/outputs/run_name"    # all outputs here
  checkpoint_dir: "checkpoints"                  # relative to base_dir
  samples_dir:    "samples"                      # relative to base_dir
```

### 3. Validate dataset

```bash
# Reads paths from config - single source of truth
python scripts/prepare_dataset.py --config configs/dgx_train.yaml

# Create dummy data for smoke testing, then validate:
python scripts/prepare_dataset.py --config configs/laptop_debug.yaml --create_dummy
```

### 4. Verify on laptop (4 GB GPU)

```bash
python train.py --config configs/laptop_debug.yaml
```

### 5. Train on DGX (16 GB GPU)

```bash
python train.py --config configs/dgx_train.yaml
```

### 6. Resume training

```bash
# Resume from last periodic checkpoint:
python train.py --config configs/dgx_train.yaml --resume last

# Resume from best-metric checkpoint:
python train.py --config configs/dgx_train.yaml --resume best

# Resume from explicit path:
python train.py --config configs/dgx_train.yaml \
    --resume /path/to/outputs/run_name/checkpoints/checkpoint-last/
```

### 7. Generate synthetic paired dataset

```bash
python inference.py \
    --checkpoint /path/to/outputs/run_name/final/model_final.pt \
    --input_dir  /path/to/normal/frames \
    --output_dir /path/to/synthetic_dataset \
    --cfg_scale  7.0 \
    --num_steps  50 \
    --image_size 256 \
    --fp16
```

Output structure:
```
synthetic_dataset/
  normal/    <- ground truth (copies of input)
  over/      <- synthetic overexposed (paired)
  under/     <- synthetic underexposed (paired)
```

---

## Config Reference

### `data` section

| Key | Description |
|---|---|
| `normal_path` | Directory of normal frames (ground truth) |
| `over_path` | Directory of overexposed reference frames (unpaired) |
| `under_path` | Directory of underexposed reference frames (unpaired) |
| `image_size` | Resize all frames to this square size (128 laptop, 256 DGX) |
| `num_workers` | DataLoader workers (2 laptop, 8 DGX) |
| `pin_memory` | false laptop, true DGX |

### `output` section

| Key | Description |
|---|---|
| `base_dir` | Root directory for all training outputs |
| `checkpoint_dir` | Subdirectory for checkpoints (relative to `base_dir`) |
| `samples_dir` | Subdirectory for sample grids (relative to `base_dir`) |

Only two checkpoint files exist on disk at any time:
- `checkpoint-last/checkpoint.pt` — overwritten every `save_every` steps
- `checkpoint-best/checkpoint.pt` — overwritten when val metric improves

### `training.best_metric`

| Value | Direction | Description |
|---|---|---|
| `exposure_visibility` | higher | Fraction of images with correct visible exposure |
| `ssim_vs_normal` | higher | Structural similarity to normal frame |
| `hist_kl_div` | lower | KL divergence from reference exposure histogram |

---

## Evaluation Metrics

| Metric | Target | Meaning |
|---|---|---|
| **EVS** (Exposure Visibility Score) | toward 1.0 | Generated images have visible, correct exposure level |
| **SSIM vs Normal** | 0.7-0.9 | Structure preserved; some exposure difference expected |
| **PSNR vs Normal** | 20-30 dB | We want difference - too high means no exposure change |
| **Luminance mean** | OVER >0.6, UNDER <0.4 | Direct brightness check |
| **Histogram KL** | toward 0 | Output distribution matches reference exposure |